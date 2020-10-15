import base64
import datetime
import json
import urllib

import requests

from settings import *
import os
from flask import Flask, request, jsonify

from utils.autostarter import AutoStarter

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
ast = AutoStarter(NN_TIMEOUT, ['python', 'xray_processing/main_utils.py'])


def allowed_file(filename):
    return '.' in filename and filename.split('.')[
        -1].lower() in ALLOWED_EXTENSIONS


def secure(filename):
    for sep in os.path.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, "_")
    filename = str("_".join(filename.split())).strip("._")
    return filename


@app.route('/process', methods=['POST'])
def process_image():
    filename = request.form['filename']
    filename = str(datetime.datetime.now()) + '_' + secure(filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    ret_val = {'error': None}
    to_delete = []

    try:
        if 'image' in request.files:
            file = request.files['image']  # File
            if not file or not allowed_file(file.filename):
                return jsonify({'error': 'not an image'})

            file.save(input_path)
        elif 'image' in request.form:
            with open(input_path, 'wb') as f:
                try:
                    r = requests.get(request.form['image'])
                    f.write(r.content)
                except Exception as e:
                    r = urllib.request.urlopen(request.form['image'])
                    f.write(r.file.read())

        else:
            raise Exception('No image were sent')

        to_delete.append(input_path)

        reply = ast.send_recv(input_path)
        print(reply)
        if reply != 'SUCCESS':
            raise Exception(reply)

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


@app.route('/examples', methods=['GET'])
def examples():
    return jsonify(sorted(os.listdir(EXAMPLES_FOLDER)))

@app.route('/examples/<path:filename>')
def get_preview(filename):
    path = os.path.join(EXAMPLES_FOLDER, secure(filename))
    try:
        with open(path, 'rb') as file:
            b64 = base64.b64encode(file.read()).decode('utf-8')
            resp = {'name': filename, 'base64': b64}
    except Exception as e:
        resp = {'error': str(e)}
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
