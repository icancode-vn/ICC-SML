
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
