from imutils import paths
import argparse
import pickle
import cv2
import os

import numpy as np
import cv2
from face_operator import  face_re
import time
cap = cv2.VideoCapture("/home/aioz-nam/Code/HC/data/blink.mp4")
face_module = face_re()
# huynh_lap = cv2.imread("/home/aioz-nam/Code/HC/data/test.png")
# cap_add = cv2.VideoCapture("/home/aioz-nam/Code/HC/data/huynhlap5s.mp4")
# while(True):
#     ret, frame = cap_add.read()
#     if(ret):
#         face_module.add_face_2(huynh_lap, "huynh lap")
#     else:
#         break
# face_module.save_face_data()
i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    # if (i % 100 != 0):
    #     i+=1
    #     continue
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    begin = time.time()
    success, boxes, predictions, face_status, yaw = face_module.re_face(frame)
    if(success):
        print("FPS: ", 1/(time.time() - begin), face_status, yaw)
        frame = face_module.draw_rect(frame, boxes, predictions, False)
    # if(not success):
    #     frame = face_module.draw_rect(frame, result)
    # # Display the resulting frame
    cv2.imshow('Video',frame)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.imwrite("test.png", frame)