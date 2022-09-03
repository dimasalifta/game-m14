import cv2
import numpy as np
import os
import time
import json
import requests
from datetime import datetime
import pyautogui
import imutils
net = cv2.dnn.readNet('yolov3_training_20000.weights', 'yolov3-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
result_value_persent = 100

def sendingData():
	dt = datetime.now()
	ts = datetime.timestamp(dt)
	myScreenshot = pyautogui.screenshot()
	imgName = str(dt) + ".jpg"
	path = 'screenshoot/'
	cv2.imwrite(os.path.join(path , imgName), roi)
	imgFile = open("screenshoot/" + imgName, "rb")
	dataJson = {
	"machineName": "m14",
	"result": data,
	"accuracy": confidence
	}
	fileImage = {
	"Image": imgFile
	}
	#print(imgName)
	res_server = requests.post(url, files = fileImage, data = dataJson)
	#print(res_server.status_code)
	print(res_server.json())
	os.remove("screenshoot/" + imgName)
def zoom_center(cap, zoom_factor):

    y_size = cap.shape[0]
    x_size = cap.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    cap_cropped = cap[y1:y2,x1:x2]
    return cv2.resize(cap_cropped, None, fx=zoom_factor, fy=zoom_factor)

url = 'http://palapa.spacearcade.online/submitresult'
old_value = None

while True:
    _, img = cap.read()
    
    #create ROI
    center_coordinates = (320, 240)   # Center coordinates 640x480
    radius = 100            # Radius of circle
    colorb = (255, 0, 0)     # Blue color in BGR
    colorr = (0, 0, 255)     # Blue color in BGR
    colorg = (0, 255, 0)
    thickness = 2           # Line thickness of 2 px
    xx1 = 200
    yy1 = 200
    xx2 = 480
    yy2 = 280

    
    cv2.rectangle(img,(xx1,yy1),(xx2,yy2),colorb,thickness)
    roi = img[yy1:yy2,xx1:xx2]
    height, width, _ = roi.shape
    zoom = zoom_center(roi,4)                               # zoom 4x ROI
    
    blob = cv2.dnn.blobFromImage(roi, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    daftar = []
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            daftar.append(label)
            data = str(daftar)
            confidence = str(round(confidences[i],2)*100)
            color = colors[i]
            cv2.rectangle(roi, (x,y), (x+w, y+h), color, 2)
            cv2.putText(roi,label, (x,y+40), font, 1, (0,255,0), 2)
            state = 0
            state += 1
            if len(daftar) < 1:
                state = 0
                print("No Dice")
            if len(daftar) == 5:
                # ================================================
                new_value = data
                state = 0
                if new_value != old_value:
                    old_value = new_value
                    #==================================================================
                    sendingData()
                    # ==================================================================
            sentStatusJSON = True
                    
    print(daftar)
    cv2.imshow('Image', img)
    cv2.imshow('roi',roi)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
