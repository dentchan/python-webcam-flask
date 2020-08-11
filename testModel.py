# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:49:25 2020

@author: rober
"""

from keras.models import load_model
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
###read test_img
import cv2
from efficientnet.keras import EfficientNetB3

test_dir = 'output/img'
test_images = []
print(os.listdir(test_dir))

for file in os.listdir(test_dir):
    test_image = cv2.imread(test_dir+'/'+file)
    test_image = cv2.resize(test_image,(100,100))
    test_images.append(test_image)
test_images=np.array(test_images)   
#print(emotion.shape)
#emotion = emotion.reshape(emotion.shape[0],100*100*3).astype('float32')/255
#print(emotion.shape)



# ### get test_label from csv
# test_label = pd.read_csv('D:/GP/dataset/EmoLabel/test.csv')
# #print(train_label.head(5))
# test_labels = []
# labels = []
# labels = test_label['name'][0:]
# for label in labels[0:]:
#     label = label[14:15]
#     test_labels.append(label)
# #print(label) #for debug
# test_labels = to_categorical(test_labels)

model=load_model("outputModel/eff.h5")



# testLoss, testAcc = model.evaluate(test_images, test_labels)#evaluate 測驗它
# print("testLoss ", testLoss)
# print("testAcc: ", testAcc)


print(test_images.shape)
# images.reshape(2,100,100,3)
predict = model.predict_classes(test_images)
predict_percent = model.predict_proba(test_images)

# print(predict[5])
# print(predict[6])
# for i in predict :
#     print(i)
num = len(predict)
for i in range(num) :
    if predict[i] == 1 :
        print("image%d"%i +" is Surprise" )
    elif predict[i] == 2 :
        print("image%d"%i +" is Fear" )
    elif predict[i] == 3 :
        print("image%d"%i +" is Disgust" )
    elif predict[i] == 4 :
        print("image%d"%i +" is Happiness" )
    elif predict[i] == 5 :
        print("image%d"%i +" is Sadness" )
    elif predict[i] == 6 :
        print("image%d"%i +" is Anger" )
    else :
        print("image%d"%i +" is Neutral" )
print(predict)
print(predict_percent)
'''
label
#1: Surprise
#2: Fear
#3: Disgust
#4: Happiness
#5: Sadness
#6: Anger
#7: Neutral
'''
