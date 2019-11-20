import base64
import datetime
import io
import time

import cv2
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from flask import Flask, request, send_from_directory, send_file, jsonify, make_response

FRONTEND_URL = 'http://localhost:9000'
ALLOWED_EXTENSIONS = {'bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm', 'ppm', 'sr',
                      'ras', 'tiff',
                      'tif'}
UPLOAD_FOLDER = 'upload_folder'
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
cors = CORS(app, resources={r"/process": {"origins": FRONTEND_URL}})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if file and allowed_file(file.filename):
        ret_val = {'error': None}
        try:
            filename = str(datetime.datetime.now()) + '_' + secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            image = cv2.imread(input_path)
            cv2.imwrite(input_path + '_.png', image)

            with open(input_path + '_.png', 'rb') as file:
                ret_val['image'] = base64.b64encode(file.read()).decode('utf-8')

        except Exception as e:
            ret_val['error'] = str(e)
        finally:
            os.remove(os.path.join(UPLOAD_FOLDER, filename))
            os.remove(os.path.join(UPLOAD_FOLDER, filename + '_.png'))
        return jsonify(ret_val)
    else:
        return jsonify({'error': 'not an image'})


if __name__ == '__main__':
    app.run()
