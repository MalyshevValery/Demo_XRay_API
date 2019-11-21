import os
import pickle
import numpy as np
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from xray_processing.vgg16_multilabel import VGG16 as VGG16m


from xray_processing.prediction_settings import XrayPredictionSettings


class ModelsLoader:
    corrs = None
    map_layer_model = None
    desc_layer_model = None
    cls_model = None
    segm_model = None
    classifiers = None

    @staticmethod
    def parse_model_type(model_type_name, multi_label):
        if model_type_name == 'VGG16':
            return VGG16m if multi_label else VGG16
        else:
            print('Unknown net model: ' + model_type_name)
            exit(1)
            return None

    def load_models(self, prediction_settings, multi_label=False):
        s: XrayPredictionSettings = prediction_settings

        if not multi_label:
            num_classes = 2
        else:
            num_classes = len(s.class_names)

        model_type = self.parse_model_type(s.model_type, multi_label)
        cls_model = model_type(weights=None, include_top=True, input_shape=(s.image_sz, s.image_sz, 1),
                               classes=num_classes)
        map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.map_layer_name).output)
        if not multi_label:
            desc_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.desc_layer_name).output)
        else:
            desc_layer_model = None

        corrs_path = s.weights_path[:-5] + '_' + s.map_layer_name + '_train_corrs.txt'
        print('Loading corrs from ' + corrs_path)
        corrs = np.loadtxt(corrs_path)
        corrs[np.isnan(corrs)] = 0
        corrs = np.sign(corrs) * np.square(corrs)

        print('Loading weights from ' + s.weights_path)
        cls_model.load_weights(s.weights_path)

        print('Loading segmentation model from ' + s.segm_model_path)
        segm_model = load_model(s.segm_model_path)

        if not multi_label:
            classifiers = {}
            print('Loading trained classifiers from ' + s.classifiers_dir)
            for class_name in s.class_names:
                path = os.path.join(s.classifiers_dir, 'logit-%s.pickle' % class_name)
                classifier = pickle.load(open(path, 'rb'))
                classifiers[class_name] = classifier
        else:
            classifiers = None

        self.cls_model = cls_model
        self.map_layer_model = map_layer_model
        self.desc_layer_model = desc_layer_model
        self.corrs = corrs
        self.segm_model = segm_model
        self.classifiers = classifiers

        return self
