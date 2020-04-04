import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Conv2D

from xray_processing.prediction_settings import XrayPredictionSettings
from xray_processing.conv2dblock import Conv2DBlock


class ModelsLoader:
    def __init__(self):
        self.corrs = None
        self.coefs = None
        self.map_layer_model = None
        self.cls_model = None
        self.segm_model = None
        self.working_points = None

    def load_models(self, prediction_settings):
        s: XrayPredictionSettings = prediction_settings

        cls_model = self._build_classification_model(s.job, (0, s.image_sz, s.image_sz, s.channels))

        corrs = None
        coefs = None
        if s.heatmap_settings.method == 'layer':
            map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.layers[-2].output)
        elif s.heatmap_settings.method == 'gradcam':
            map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.map_layer_name).output)
        elif s.heatmap_settings.method == 'corrs':
            map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.map_layer_name).output)
            corrs_path = s.weights_path[:-5] + '_' + s.map_layer_name + '_max_train_corrs.txt'
            print('Loading corrs from ' + corrs_path)
            corrs = np.loadtxt(corrs_path)
            corrs[np.isnan(corrs)] = 0
            corrs = np.sign(corrs) * np.square(corrs)
        elif s.heatmap_settings.method == 'coefs':
            map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.map_layer_name).output)
            coefs_path = s.weights_path[:-5] + '_' + s.map_layer_name + '_max_train_coefs.txt'
            print('Loading coefs from ' + coefs_path)
            coefs = np.loadtxt(coefs_path)[:-1]
            coefs[np.isnan(coefs)] = 0
        else:
            raise Exception('Unsupported heatmap method "%s"' % s.heatmap_settings.method)

        print('Loading weights from ' + s.weights_path)
        cls_model.load_weights(s.weights_path)

        print('Loading segmentation model from ' + s.segm_model_path)
        segm_model = load_model(s.segm_model_path, compile=False, custom_objects={'Conv2DBlock': Conv2DBlock})

        if s.normalize_scores:
            points_path = os.path.join(s.job_dir, 'working_points_val_minloss.json')
            if os.path.isfile(points_path):
                with open(points_path, 'r') as f:
                    self.working_points = json.load(f)
            else:
                print('*** WARNING: Score normalization is set to "True" but file not found at "%s"' % points_path)
                print('Scores normalization will not be performed!')

        self.cls_model = cls_model
        self.map_layer_model = map_layer_model
        self.corrs = corrs
        self.coefs = coefs
        self.segm_model = segm_model

        return self

    @staticmethod
    def _parse_model(model_type):
        if model_type == 'InceptionV3':
            return InceptionV3
        if model_type == 'VGG16':
            return VGG16
        if model_type == 'VGG19':
            return VGG19
        if model_type == 'ResNet50':
            return ResNet50
        if model_type == 'InceptionResNetV2':
            return InceptionResNetV2
        if model_type == 'MobileNet':
            return MobileNet
        else:
            print('Unknown net model: ' + model_type)
            return None

    def _build_classification_model(self, job, x_shape):
        model_type = self._parse_model(job['model'])

        num_classes = 1 if job['binary'] else len(job['labels'])
        input_shape = (x_shape[1], x_shape[2], x_shape[3])
        base_model = model_type(weights=None, include_top=False, input_shape=input_shape)

        if job['pooling'] in ['avg', 'max', 'flt']:
            x = base_model.output

            if job['pooling'] == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif job['pooling'] == 'max':
                x = GlobalMaxPooling2D()(x)
            elif job['pooling'] == 'flt':
                x = Flatten()(x)

            for fc_id in range(job['fc_num']):
                x = Dense(job['fc_size'], activation='relu', name='fc%i' % fc_id)(x)

            predictions = Dense(num_classes, activation='sigmoid')(x)
        elif job['pooling'] in ['havg', 'hmax']:
            x = base_model.output

            for fc_id in range(job['fc_num']):
                x = Conv2D(job['fc_size'], (1, 1), activation='relu', name='pxw%i' % fc_id)(x)

            x = Conv2D(num_classes, (1, 1), activation='sigmoid', name='heatmaps')(x)
            if job['pooling'] == 'hmax':
                predictions = GlobalMaxPooling2D()(x)
            else:
                predictions = GlobalAveragePooling2D()(x)
        else:
            raise Exception('Unsupported pooling ' + job['pooling'])

        model = Model(inputs=base_model.input, outputs=predictions)
        return model
