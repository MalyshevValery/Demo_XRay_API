import base64
import datetime
import json
import multiprocessing
import subprocess

from process import predict_single_image
from settings import *
import cv2
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from flask import Flask, request, jsonify

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
        to_delete = []
        try:
            filename = str(datetime.datetime.now()) + '_' + secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)
            to_delete.append(input_path)

            if not input_path.endswith('.png'):
                image = cv2.imread(input_path)
                cv2.imwrite(input_path + '_.png', image)
                to_delete.append(input_path + '_.png')
                input_path = input_path + '_.png'

            #predict_single_image(input_path)
            code = subprocess.call(["python", "process.py", input_path])

            with open(input_path + '_proc.png', 'rb') as file:
                ret_val['proc'] = base64.b64encode(file.read()).decode('utf-8')
                to_delete.append(input_path + '_proc.png')

            with open(input_path + '_norm.png', 'rb') as file:
                ret_val['norm'] = base64.b64encode(file.read()).decode('utf-8')
                to_delete.append(input_path + '_norm.png')

            with open(input_path + '_data.json', 'r') as file:
                obj = json.load(file)
                ret_val['data'] = obj
                to_delete.append(input_path + '_data.json')

        except Exception as e:
            ret_val['error'] = str(e)
        finally:
            for s in to_delete:
                os.remove(s)
        return jsonify(ret_val)
    else:
        return jsonify({'error': 'not an image'})


if __name__ == '__main__':
    app.run()
