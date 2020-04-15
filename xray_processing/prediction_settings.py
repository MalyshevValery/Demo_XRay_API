import os
import json


class XrayPredictionSettings:
    def __init__(self, setup_file_path):
        print('Loading setup from ' + setup_file_path)
        with open(setup_file_path, 'r') as f:
            setup = json.load(f)

        job_file_path = os.path.join(setup['job_dir'], 'job.json')
        print('Loading job from ' + job_file_path)
        with open(job_file_path, 'r') as f:
            job = json.load(f)

        self.image_sz = job['image_size']
        self.job_dir = setup['job_dir']
        self.job = job
        self.weights_path = os.path.join(self.job_dir, 'weights-min_val_loss.hdf5')
        self.map_layer_name = setup['map_layer_name']
        self.model_type = job['model']
        self.class_names = job['labels']
        self.normalization = setup['normalization']
        self.normalize_scores = setup['normalize_scores']
        self.to_noise = setup['to_noise']
        self.segm_model_path = setup['segm_model_path']
        self.channels = job['channels'] if 'channels' in job else 1
        self.heatmap_settings = HeatMapSettings(setup['heatmap'])
        self.background_saturation = setup['background_saturation']
        self.use_crutch = setup['use_crutch']


class HeatMapSettings:
    METHODS = ['layer', 'gradcam', 'corrs', 'coefs']

    def __init__(self, settings_dict):
        if settings_dict['method'] not in self.METHODS:
            raise Exception('Unknown heatmap method "%s", must be one of %s' % (settings_dict['method'], self.METHODS))

        self.method = settings_dict['method']
        self.layer_channel = settings_dict['layer_channel']
        self.gradcam_multiplier = settings_dict['gradcam_multiplier']
        self.corrs_multiplier = settings_dict['corrs_multiplier']
        self.coefs_multiplier = settings_dict['coefs_multiplier']


if __name__ == '__main__':
    XrayPredictionSettings('setup_vgg16h_2.json')
