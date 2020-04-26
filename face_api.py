
from flask import Flask, request
from flask_socketio import SocketIO
import os
from werkzeug import secure_filename, SharedDataMiddleware
import base64
from server_utils import decode_image
from flask import jsonify

from face_operator import  face_re


app = Flask(__name__)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {'/': os.path.join(os.path.dirname(__file__), 'static')})
app.config["CACHE_TYPE"] = "null"
socketio = SocketIO(app)

face_module = face_re()


def add_face(img, name):

        res = {}
        face_module.add_face_2(img, name)
        res['success'] = True
        res['error'] = "None"
        face_module.save_face_data()
        return res
def face_processing(img):
    res = {}
    success, boxes, predictions, face_status, yaw, emotion_text = face_module.re_face(img)
    res['success'] = True
    res['boxes'] = boxes
    res['face_status'] = face_status
    res['yaw'] = (yaw=="Yawning")
    res['emotion_text'] = emotion_text
    return res



@app.route('/face_module', methods=['POST'])
def face_module():
    if request.method == 'POST':
        module_type = request.form['type']
        # type 1 add face
        # type 2 face processing
        img_64 = request.files['img'].read()
        img = decode_image(img_64, 1)
        if(module_type == 1):
            name = request.form['name']
            res = add_face(img, name)
        elif(module_type == 2):
            res = face_processing(img)

    return jsonify(res)

# type 1 add face
# type 2 face processing
if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=8000,
        debug=False,
        use_reloader=False,
        log_output=False,
        # keyfile='privkey.pem', certfile='fullchain.pem'
    )
