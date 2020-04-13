from settings import CONFIG_PATH, GPU, SOCKET_NAME
import os
import json
import numpy as np
from skimage import io, transform
from utils.autolistener import AutoListener
from xray_processing.xray_predictor import XrayPredictor

os.environ["CUDA_VISIBLE_DEVICES"] = GPU


def pred2str(predictions, items_per_row=3):
    rows = []

    i = 0
    row = ''
    for class_name in predictions:
        row += class_name + '=' + str(predictions[class_name])
        i += 1

        if i % items_per_row != 0:
            row += ', '
        else:
            rows.append(row)
            row = ''

    if row:
        rows.append(row)

    return '\n'.join(rows)


def save_combined(img_normalized, image_path, predictions, rgb, xp):
    tmp = rgb * 0
    for c in range(3):
        tmp[:, :, c] = img_normalized

    io.imsave(image_path + '_norm.png', tmp)
    io.imsave(image_path + '_proc.png', rgb)

    hm = transform.resize(xp.heat_map, xp.mask.shape)
    predictions['heat_max'] = np.max(hm)
    predictions['heat_mean'] = np.mean(hm)

    hm *= xp.mask
    predictions['lung_heat_max'] = np.max(hm)
    predictions['lung_heat_mean'] = np.mean(hm)

    with open(image_path + '_data.json', 'w') as f:
        json.dump(predictions, f, indent=2)


def predict_single_image(image_path, xp):
    predictions, rgb, img_normalized = xp.load_and_predict_image(image_path)
    save_combined(img_normalized, image_path, predictions, rgb, xp)


if __name__ == '__main__':
    xp = XrayPredictor(CONFIG_PATH, cpu_only=GPU == '')

    alist = AutoListener(SOCKET_NAME,
                         lambda input_: predict_single_image(image_path=input_,
                                                             xp=xp))
    print('Predictor process is dead')
