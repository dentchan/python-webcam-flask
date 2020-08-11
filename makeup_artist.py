from PIL import Image
import os
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time
from builtins import globals

class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # grab the frame dimensions and convert it to a blob
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
        if confidence < 0.5:  
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
                            # draw the bounding box of the face along with the associated
                # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY),
                    (0, 255, 255), 2)
        #       cv2.putText(frame, text, (startX, y),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
            roi = gray[startY:startY + endY, startX:startX + endX]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            cut_img=img[startY:endY,startX:endX]
        return cut_img
