
import base64
import requests
import cv2
import numpy
import os
from datetime import datetime

UPLOAD_FOLDER = './static/uploaded/'

### Utils funcs
def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def base64_decode(s):
    s = str(s).strip()
    try:
        return base64.b64decode(s)
    except TypeError:
        padding = len(s) % 4
        if padding > 1:
            s += b'='* (4 - padding)
            return base64.b64decode(s)
        else:
            return -1

def decode_image(image_str, type_decode=0):
    # Base64
    if type_decode == 0:
        image_data = base64_decode(image_str)
        if isinstance(image_data, int):
            return None
        else:
            try:
                img = cv2.imdecode(numpy.fromstring(image_data, numpy.uint8), cv2.IMREAD_ANYCOLOR)
                return img
            except Exception as e:
                print('Decode base64 image error: ', e)
                return None
    # Bytes
    else:
        try:
            npimg = numpy.fromstring(image_str, numpy.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print('Decode bytes image error: ', e)
            return None


def check_key_cmnd_upload(form):
    if 'front' in form and 'back' in form:
        return 2
    elif 'front' in form:
        return 1
    elif 'back' in form:
        return 0
    else:
        return -1


def save_img(img, folder, prefix=''):
    name_save = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    folder_path = UPLOAD_FOLDER + folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    if prefix != '':
        img_path = os.path.abspath(folder_path + '/' + name_save + prefix + '.jpg')
    else:
        img_path = os.path.abspath(UPLOAD_FOLDER + prefix + name_save + '.jpg')
    cv2.imwrite(img_path, img)
    print('Image saved at ', img_path)
    return img_path


def check_content_type_file(file):
    content_type = file.content_type.split('/')[0]
    return content_type == 'image'