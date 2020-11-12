import os
import tensorflow as tf
import pydicom
import numpy as np
import tensorflow
from skimage import io, color, exposure, morphology
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from xray_processing.prediction_settings import XrayPredictionSettings
from xray_processing.models_loader import ModelsLoader
from .cropping import Cropping
from xray_processing.imutils import imresize, normalize_by_lung_convex_hull, normalize_by_lung_mean_std


class XrayPredictor:
    def __init__(self, setup_file_path, cpu_only=False, gpu_list='0'):
        self._set_tf_session(cpu_only, gpu_list)
        self.prediction_settings = XrayPredictionSettings(setup_file_path)
        self.models = ModelsLoader().load_models(self.prediction_settings)
        self.img_original = None
        self.mask = None
        self.img_roi = None
        self.heat_map = None

    def load_and_predict_image(self, input_image_path):
        self.img_original = self._load_original_image(input_image_path)

        img_gray = self._convert_to_gray(self.img_original)

        preview = self._make_preview(img_gray)

        lungs = self._segment_lungs(preview)

        img_normalized, mask, img_roi, mask_roi, cropping = self._normalize_and_crop(img_gray, lungs)

        heat_map, prob, predictions = self._infer_neural_net(img_roi)

        rgb = self._make_colored(img_normalized, mask, heat_map, cropping)

        self.mask = mask
        self.heat_map = heat_map
        self.img_roi = img_roi
        return predictions, rgb, img_normalized

    @staticmethod
    def _load_original_image(input_image_path):
        # print('Loading image from ' + input_image_path)

        ext = os.path.splitext(input_image_path)[-1].lower()
        if ext in ['.jpg', '.png', '.bmp', '.jpeg']:
            img_original = io.imread(input_image_path)
        elif ext in ['.dcm', '.dicom', '.bin', '']:
            dcm = pydicom.dcmread(input_image_path, force=True)
            img_original = dcm.pixel_array
            if ext == '.bin':
                img_original = np.max(img_original) - img_original
                img_original = img_original[:, ::-1]
            # if 'PhotometricInterpretation' in dcm.dir() and dcm.PhotometricInterpretation.upper() == 'MONOCHROME2':
            #     img_original = np.max(img_original) - img_original
            # if 'ViewPosition' in dcm.dir() and dcm.ViewPosition.upper() == 'AP':
            #     img_original = img_original[:, ::-1]
        elif ext in ['.eli']:
            img_original = XrayPredictor._load_eli_image(input_image_path)
        else:
            raise Exception('Unsupported input image extension: ' + ext)

        # print('Loaded image (%i x %i)' % (img_original.shape[0], img_original.shape[1]))
        return img_original

    @staticmethod
    def _load_eli_image(input_image_path):
        with open(input_image_path, 'rb') as f:
            all_bytes = f.read()

        resolution_bytes = all_bytes[16:24]
        resolution = np.frombuffer(resolution_bytes, dtype=np.uint32)
        num_pixels = np.prod(resolution)

        pixel_bytes = all_bytes[-int(num_pixels * 2):]
        pixels = np.frombuffer(pixel_bytes, dtype=np.int16)

        w, h = tuple(resolution)
        img_original = pixels.reshape((h, w))
        img_original = 2**15 - img_original

        return img_original

    @staticmethod
    def _convert_to_gray(img_original):
        if len(img_original.shape) > 2:
            if img_original.shape[2] == 1:
                img_gray = img_original[:, :, 0].copy()
            elif img_original.shape[2] == 3:
                img_gray = color.rgb2gray(img_original)
            elif img_original.shape[2] == 4:
                img_gray = color.rgb2gray(img_original[:, :, 0:3])
            else:
                raise Exception('Unsupported number of channels of the input image: ' + img_original.shape[2])
        else:
            img_gray = img_original.copy()

        img_gray = img_gray.astype(np.float32)

        return img_gray

    @staticmethod
    def _make_preview(img_gray):
        preview_size = 256

        preview = imresize(img_gray, (preview_size, preview_size))
        preview = exposure.equalize_hist(preview)

        preview[preview < 0] = 0
        preview[preview > 1] = 1

        return preview

    def _segment_lungs(self, preview):
        def remove_small_regions(img, size):
            img = morphology.remove_small_objects(img, size)
            img = morphology.remove_small_holes(img, size)
            return img

        x = preview.copy()

        # TODO make adjustable through config
        # x -= x.mean()
        # x /= x.std()
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)

        segm_model = self.models.segm_model
        lungs = segm_model.predict(x, batch_size=1)[..., 0].reshape(preview.shape)
        lungs = lungs > 0.5
        lungs = remove_small_regions(lungs, 0.02 * np.prod(preview.shape))

        return lungs

    @staticmethod
    def _get_cropping(mask_bw):
        proj_x = np.sum(mask_bw, axis=0).flatten()
        proj_y = np.sum(mask_bw, axis=1).flatten()

        d = min(mask_bw.shape) // 20
        x_low = max(0, np.where(proj_x > 0)[0][0] - d)
        x_high = min(mask_bw.shape[1], np.where(proj_x > 0)[0][-1] + d)
        y_low = max(0, np.where(proj_y > 0)[0][0] - d)
        y_high = min(mask_bw.shape[0], np.where(proj_y > 0)[0][-1] + d)

        return Cropping(x_low, x_high, y_low, y_high)

    @staticmethod
    def _put_noise(img, mask):
        lung_intensities = img.flatten()
        lung_intensities = lung_intensities[mask.flatten() > 0.5]
        mean_intensity = np.mean(lung_intensities)
        std_intensity = np.std(lung_intensities)

        noise = np.random.normal(mean_intensity, std_intensity, img.shape)
        noised = img * mask + noise * (1 - mask)
        return noised

    def _normalize_and_crop(self, img_gray, lungs):
        image_sz = self.prediction_settings.image_sz

        mask = imresize(lungs, img_gray.shape, order=0)

        if self.prediction_settings.normalization == 'conv_hull':
            img_normalized = normalize_by_lung_convex_hull(img_gray, mask)
        elif self.prediction_settings.normalization == 'mean_std':
            img_normalized = normalize_by_lung_mean_std(img_gray, mask)
        else:
            raise Exception('Unknown normalization: ' + str(self.prediction_settings.normalization))

        img_normalized[img_normalized < 0] = 0
        img_normalized[img_normalized > 1] = 1

        if self.prediction_settings.to_noise:
            img_prepared = self._put_noise(img_normalized, mask)
        else:
            img_prepared = img_normalized

        cropping = self._get_cropping(mask)
        img_roi = cropping.crop_image(img_prepared)
        mask_roi = cropping.crop_image(mask)

        img_roi = imresize(img_roi, (image_sz, image_sz), order=1)
        mask_roi = imresize(mask_roi, (image_sz, image_sz), order=0)

        return img_normalized, mask, img_roi, mask_roi, cropping

    def _infer_neural_net(self, img_roi):
        m: ModelsLoader = self.models
        s: XrayPredictionSettings = self.prediction_settings

        image_sz = img_roi.shape[0]

        x = img_roi - 0.5
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)

        # print('Evaluating net')
        meta = np.array([[0.5, 1]]).astype(x.dtype)
        prob = m.cls_model.predict([x, meta], batch_size=1)[0, :]

        if s.use_crutch and s.heatmap_settings.method == 'layer':
            map_layer_output = m.map_layer_model.predict([x, meta], batch_size=1)
            map_layer_output[0, 2:5, 4:6, :] = 0
            prediction_scores = np.max(map_layer_output[0], axis=(0, 1))
            predictions = dict()
            for i, class_name in enumerate(s.class_names):
                predictions[class_name] = round(prediction_scores[i], 3)

            heat_map = map_layer_output[0, :, :, 0].astype(float)
            heat_map = imresize(heat_map, (image_sz, image_sz))
        else:
            predictions = self._compose_predictions(prob)
            heat_map = self._build_heatmap(image_sz, x, meta, predictions['class_number'])

        return heat_map, prob, predictions

    def _build_heatmap(self, image_sz, x, meta, max_prob):
        m: ModelsLoader = self.models
        s: XrayPredictionSettings = self.prediction_settings

        if s.heatmap_settings.method == 'corrs':
            map_layer_output = m.map_layer_model.predict([x, meta], batch_size=1)
            heat_map = np.matmul(map_layer_output, m.corrs)[0, ...]
            heat_map = imresize(heat_map, (image_sz, image_sz))
            # heat_map *= s.heatmap_settings.corrs_multiplier
            heat_map *= max_prob / heat_map.max()
        elif s.heatmap_settings.method == 'coefs':
            map_layer_output = m.map_layer_model.predict([x, meta], batch_size=1)
            heat_map = np.matmul(map_layer_output, m.coefs)[0, ...]
            heat_map = imresize(heat_map, (image_sz, image_sz))
            # heat_map *= s.heatmap_settings.coefs_multiplier
            heat_map *= max_prob / heat_map.max()
        elif s.heatmap_settings.method == 'layer':
            map_layer_output = m.map_layer_model.predict([x, meta], batch_size=1)
            heat_map = map_layer_output[0, :, :, 0].astype(float)
            heat_map = imresize(heat_map, (image_sz, image_sz))
        elif s.heatmap_settings.method == 'gradcam':
            heat_map = self._build_heatmap_grad_cam([x, meta], max_prob)
        else:
            raise Exception('Unsupported heatmap method "%s"' % s.heatmap_settings.method)

        heat_map[heat_map < 0] = 0
        heat_map[heat_map > 1] = 1
        return heat_map

    def _build_heatmap_grad_cam(self, x, max_prob):
        m: ModelsLoader = self.models
        s: XrayPredictionSettings = self.prediction_settings

        model = m.cls_model

        grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(s.map_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            (layer_outputs, model_output) = grad_model([x])
            grads = tape.gradient(model_output[:, 0], layer_outputs)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = np.matmul(layer_outputs, pooled_grads)[0]
        heatmap[heatmap < 0] = 0
        if np.max(heatmap) > 0:
            heatmap *= max_prob / np.max(heatmap) * 1.5
            heatmap[heatmap > max_prob] = max_prob

        return heatmap

    def _compose_predictions(self, prob):
        s: XrayPredictionSettings = self.prediction_settings

        predictions = {}
        for i, class_name in enumerate(s.class_names):
            predictions[class_name] = float(prob[i])

        if self.models.working_points:
            self._normalize_prediction_scores(predictions)

        for class_name in s.class_names:
            predictions[class_name] = round(predictions[class_name], 3)

        return predictions

    def _normalize_prediction_scores(self, predictions):
        s: XrayPredictionSettings = self.prediction_settings
        wp = self.models.working_points
        fp = np.array([0, 0.25, 0.5, 0.75, 1])
        for class_name in s.class_names:
            x = predictions[class_name]
            thr0 = wp[class_name]['high_sens']['threshold']
            thr1 = wp[class_name]['balanced']['threshold']
            thr2 = wp[class_name]['high_spec']['threshold']
            xp = [0, thr0, thr1, thr2, 1]
            predictions[class_name] = np.interp(x, xp, fp, left=0, right=1)
            if class_name != 'class_number':
                predictions[class_name] *= predictions['class_number']

    def _make_colored(self, img_normalized, mask, heat_map, cropping):
        sz = img_normalized.shape
        hsv = np.zeros((sz[0], sz[1], 3))

        v = img_normalized
        v[v < 0] = 0
        v[v > 1] = 1
        hsv[:, :, 2] = 0.1 + 0.9 * v
        hsv[:, :, 1] = mask * 0.5 + self.prediction_settings.background_saturation

        x_low, x_high, y_low, y_high = cropping.unpack_values()

        map = img_normalized * 0
        map[y_low:y_high, x_low:x_high] = imresize(heat_map, (y_high - y_low, x_high - x_low))
        map[map < 0] = 0
        map[map > 1] = 1
        hsv[:, :, 0] = 0.7 * (1 - map)

        rect_hue = 0.8
        rect_sat = 1
        d = 3
        hsv[y_low:y_low + d, x_low:x_high, 0] = rect_hue
        hsv[y_high:y_high + d, x_low:x_high, 0] = rect_hue
        hsv[y_low:y_high, x_low:x_low + d, 0] = rect_hue
        hsv[y_low:y_high, x_high:x_high + d, 0] = rect_hue

        hsv[y_low:y_low + d, x_low:x_high, 1] = rect_sat
        hsv[y_high:y_high + d, x_low:x_high, 1] = rect_sat
        hsv[y_low:y_high, x_low:x_low + d, 1] = rect_sat
        hsv[y_low:y_high, x_high:x_high + d, 1] = rect_sat

        rgb = color.hsv2rgb(hsv)

        return rgb

    @staticmethod
    def _set_tf_session(cpu_only, gpu_list):
        if tensorflow.__version__.startswith('1.'):
            from tensorflow.keras.backend import set_session
            if cpu_only:
                config = tf.ConfigProto(device_count={'GPU': 0})
            else:
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.5
                config.gpu_options.visible_device_list = gpu_list
            set_session(tf.Session(config=config))


def main():
    xp = XrayPredictor('setup_vgg16_1.json', cpu_only=True)
    predictions, rgb, img_normalized = xp.load_and_predict_image('test_data/tb_01.jpg')
    print(predictions)
    io.imsave('temp_rgb.png', (rgb * 255).astype(np.uint8))
    io.imsave('temp_normalized.png', (img_normalized * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
