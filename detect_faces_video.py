# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 沒GPU時打這行可以讓他改用CPU跑
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

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# global net, emotion_classifier, vs
# =============================================================================
# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# global angry, disgust, fear, happy, sad, surprice
# angry = []; disgust = []; fear = []; happy = []; sad = []; surprice = []
# =============================================================================

def test(prototxt, model):
	# load our serialized model from disk
	print("[INFO] Face-Detect loading model...")
# 	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	net = cv2.dnn.readNetFromCaffe(prototxt, model)
	
# =============================================================================
# 	print("[INFO] Emotion-Detect Loading model...")
# 	emotion_classifier = tf.keras.models.load_model("models/Model_Original_v3.model")
# =============================================================================
	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] starting video stream...")    
	vs = VideoStream(src = 0).start()
	time.sleep(2.0)
	aa = 0
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		canvas = np.zeros((250, 300, 3), dtype="uint8")
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()
		
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]
	
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < 0.5:
				continue
	
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
	# 		cv2.putText(frame, text, (startX, y),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			
			roi = gray[startY:startY + endY, startX:startX + endX]
			roi = cv2.resize(roi, (48, 48))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)
		
# =============================================================================
# 			preds = emotion_classifier.predict(roi)[0]
# 			emotion_probability = np.max(preds)
# 			label = EMOTIONS[preds.argmax()]
# =============================================================================
			
# =============================================================================
# 			cv2.putText(frame, label, (startX, y),
# 	 		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# 			
# 		for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
# 			# construct the label text
# 			text = "{}: {:.2f}%".format(emotion, prob * 100)
# 			outPercent = "{:.2f}".format(prob * 100)
# 			if(i == 0):
# 				angry.append(float(outPercent))
# 			elif(i == 1):
# 				disgust.append(float(outPercent))
# 			elif(i == 2):
# 				fear.append(float(outPercent))
# 			elif(i == 3):
# 				happy.append(float(outPercent))
# 			elif(i == 4):
# 				sad.append(float(outPercent))
# 			elif(i == 5):
# 				surprice.append(float(outPercent))
# 			# draw the label + probability bar on the canvas
# 			# emoji_face = feelings_faces[np.argmax(preds)]
# =============================================================================
				
# =============================================================================
# 			w = int(prob * 300)
# 			cv2.rectangle(canvas, (7, (i * 35) + 5),
# 			(w, (i * 35) + 35), (0, 0, 255), -1)
# 				
# 			cv2.putText(canvas, text, (10, (i * 35) + 23),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45,
# 			(255, 255, 255), 1)
# =============================================================================
			
# 		aa += 1
# 		print("angry", angry)
# 		print("disgust", disgust)
# 		print("fear", fear)
# 		print("happy", happy)
# 		print("sad", sad)
# 		print("surprice", surprice)
# 		
# 		if(aa % 10 == 0):
# 			print(aa)
# 			threeInOne()
# 			print("angry", angry)
# 			print("disgust", disgust)
# 			print("fear", fear)
# 			print("happy", happy)
# 			print("sad", sad)
# 			print("surprice", surprice)
# 		print(aa)
# 		time.sleep(1.0)
		# show the output frame
		cv2.imshow("Frame", frame)
		# cv2.imshow("Emotion", canvas)
		key = cv2.waitKey(1) & 0xFF
		
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
	
# =============================================================================
# def threeInOne():
# 	global angry, disgust, fear, happy, sad, surprice
# 	allLen = int(len(angry) - 1)
# 	
# 	score = angry[0]
# 	for i in range(1, allLen):
# 		score = (score + angry[i])
# 	score = score/allLen
# 	angry.clear()
# 	angry.append(score)
# 	
# 	score = disgust[0]
# 	for i in range(1, allLen):
# 		score = (score + disgust[i])
# 	score = score/allLen
# 	disgust.clear()
# 	disgust.append(score)
# 	
# 	score = fear[0]
# 	for i in range(1, allLen):
# 		score = (score + fear[i])
# 	score = score/allLen
# 	fear.clear()
# 	fear.append(score)
# 	
# 	score = happy[0]
# 	for i in range(1, allLen):
# 		score = (score + happy[i])
# 	score = score/allLen
# 	happy.clear()
# 	happy.append(score)
# 	
# 	score = sad[0]
# 	for i in range(1, allLen):
# 		score = (score + sad[i])
# 	score = score/allLen
# 	sad.clear()
# 	sad.append(score)
# 	
# 	score = surprice[0]
# 	for i in range(1, allLen):
# 		score = (score + surprice[i])
# 	score = score/allLen
# 	surprice.clear()
# 	surprice.append(score)
# =============================================================================
	
test("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
