import tkinter as tk
from PIL import Image, ImageOps,ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import imutils
import numpy as np

from data import group_config as config
from data.detection import detect_people
from scipy.spatial import distance as dist
from threading import Thread
import argparse
import os
import playsound

import warnings
warnings.filterwarnings("ignore")
import time
from sort import *
from collections import deque
from analyse import *


window = tk.Tk()
window.title("Unusual_Detection")

window.geometry('800x450')
window.configure(background='snow')

message = tk.Label(window, text="Unusual Detection", bg="snow", fg="black", width=35,
		   height=2, font=('times', 26, 'italic bold '))
message.place(x=0, y=0)


ALARM_ON = False

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def audioalert():
    global ALARM_ON
    # if the alarm is not on, turn it on
    if not ALARM_ON:
	    ALARM_ON = True
	    
	    # check to see if an alarm file was supplied,
	    # and if so, start a thread to have the alarm
	    # sound played in the background
	    if "alarm.wav" != "":
		    t = Thread(target=sound_alarm,
			    args=("alarm.wav",))
		    t.deamon = True
		    t.start()




# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


		    
# load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]




def live():
	import live_video

def video():
        global path
        global vi
        
        path=askopenfilename(filetypes=[("",'')])
        vi = cv2.VideoCapture(path)
        video_1()

def group():
        global path
        global vs
        path=askopenfilename(filetypes=[("",'')])
        vs = cv2.VideoCapture(path)
        analysis()

	
def exit():
	window.destroy()

def clear():
      rst.destroy()

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
		    window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

def analysis():
        while True:
                # read the next frame from the file
                (grabbed, frame) = vs.read()

                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                        break

                # resize the frame and then detect people (and only people) in it
                frame = imutils.resize(frame, width=700)
                results = detect_people(frame, net, ln,
                        personIdx=LABELS.index("person"))

                # initialize the set of indexes that violate the minimum social
                # distance
                violate = set()

                # ensure there are *at least* two people detections (required in
                # order to compute our pairwise distance maps)
                if len(results) >= 2:
                        # extract all centroids from the results and compute the
                        # Euclidean distances between all pairs of the centroids
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                        # loop over the upper triangular of the distance matrix
                        for i in range(0, D.shape[0]):
                                for j in range(i + 1, D.shape[1]):
                                        # check to see if the distance between any two
                                        # centroid pairs is less than the configured number
                                        # of pixels
                                        if D[i, j] < config.MIN_DISTANCE:
                                                # update our violation set with the indexes of
                                                # the centroid pairs
                                                violate.add(i)
                                                violate.add(j)

                # loop over the results
                for (i, (prob, bbox, centroid)) in enumerate(results):
                        # extract the bounding box and centroid coordinates, then
                        # initialize the color of the annotation
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)
                        # if the index pair exists within the violation set, then
                        # update the color
                        if i in violate:
                                color = (0, 0, 255)
                        # draw (1) a bounding box around the person and (2) the
                        # centroid coordinates of the person,
                        cv2.putText(frame, str(i), (startX,startY-5),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 1)

                # draw the total number of social distancing violations on the
                # output frame
                text = "Social Distancing Violations: {}".format(len(violate))
                vio = len(violate)
                #print(type(vio))
                if(vio > 1):
                        audioalert()
                cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                        # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break
        vs.release()
        cv2.destroyAllWindows()
        print("[INFO Closed]")


def video_1():
        cross_check = []
        tracker = Sort()
        memory = {}
        pointsDict = {}
        TrackedIDs = []
        counter1 = 0
        writer = None
        (W, H) = (None, None)
         
        frameIndex = 0
        calculatePeopleCount = True
        calculateSpeed = True
        Id = 0
        prevPeopleCount = 0
        totalPeopleCount = 0
        totalLineCrossedLeft = 0
        totalLineCrossedRight = 0
        totalLineCrossed = 0
        lineCrossingIDs = [] #list of ID's which are currantly crossing the line
        calculateLineCrossed = True
        frames = 0
        line1 = [(500,263), (780, 263)]
        counter1 = 0 
        while True:
                peoplecount = 0
                totalSpeed = 0
                # read the next frame from the file
                (grabbed, frame) = vi.read()
                if not grabbed:
                    break
                frames += 1
                frame = imutils.resize(frame,width=800)
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416,416),swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()
            
                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                center = []
                confidences = []
                classIDs = []
            
                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
            
                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.3:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            
                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
            
                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            center.append(int(centerY))
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                            
            
                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
                #print("idxs", idxs)
                #print("boxes", boxes[i][0])
                #print("boxes", boxes[i][1])
                
                dets = []
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        dets.append([x, y, x+w, y+h, confidences[i]])
                        #print(confidences[i])
                        #print(center[i])
                np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                dets = np.asarray(dets)
                tracks = tracker.update(dets)
                
                boxes = []
                indexIDs = []
                c = []
                
                previous = memory.copy()
                #print("centerx",centerX)
                #  print("centery",centerY)
                memory = {}
            
                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    memory[indexIDs[-1]] = boxes[-1]
            
                if len(boxes) > 0:
                    i = int(0)
                    for box in boxes:
                                (x, y) = (int(box[0]), int(box[1]))
                                (w, h) = (int(box[2]), int(box[3]))
                    
                                # draw a bounding box rectangle and label on the image
                                # color = [int(c) for c in COLORS[classIDs[i]]]
                                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(y - (h-y))),int(x - (w-x))
                                 
                    
                                if indexIDs[i] in previous:
                                    previous_box = previous[indexIDs[i]]
                                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                    p1 = (int(y2 - (h2-y2))),int(x2 - (w2-x2))
                                    #cv2.line(frame, p0, p1, color, 3)
                                    #p2 = (int(10+x2 + (w2-x2)/2), int(10+y2 + (h2-y2)/2))
                                    #cv2.putText(frame, str(p1), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                                    Id = indexIDs[i]
                                    
                                if frames % 24 == 0:
                                        if intersect(p0, p1, line1[0], line1[1]) and indexIDs[i] not in cross_check:
                                                counter1 += 1
                                                if(counter1 > 2):
                                                        audioalert()
                                                #print(counter1)
                                                cross_check.append(indexIDs[i])
                                                
                                if calculatePeopleCount:
                                        peoplecount += 1
                                        if Id not in TrackedIDs:
                                                TrackedIDs.append(Id)
                                                #print(TrackedIDs)
                                #add center to dict
                                if Id in pointsDict:
                                         pointsDict[Id].appendleft(p0)
                                else:
                                        pointsDict[Id] = deque(maxlen=25)
                                        pointsDict[Id].appendleft(p0)

                                if calculateSpeed:
                                        #print(pointsDict[Id])
                                        speed = getSpeed(pointsDict[Id])
                                        try:
                                                
                                                totalSpeed += speed
                                                #print(totalSpeed)
                                                if totalSpeed == 0:
                                                        state = "Still"
                                                        color = (0,255,0)
                                                else:
                                                        state = "Moving"
                                                        color = (0,0,255)
                                        except:
                                                continue
                                cv2.line(frame, (500,263), (780,263), [0, 255, 0], 5)
                                cv2.putText(frame, "Counter: " + str(counter1), (30,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                                cv2.putText(frame, "Person Id: " + str(peoplecount), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                cv2.rectangle(frame, (x, y), (w, h), color, 4)
                                cv2.putText(frame, state, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                cv2.imshow("Image",frame)
                key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
                if key == 27: #esc key stops the process
                    break
            
        vi.release()    
        cv2.destroyAllWindows()
        
	
takeImgr = tk.Button(window, text="Live",command=live,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImgr.place(x=90, y=150)

trainImgf = tk.Button(window, text="Video",fg="black",command=video,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImgf.place(x=400, y=150)

takeImg = tk.Button(window, text="Group Detection",command=group,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=300)

quitWindow = tk.Button(window, text="Quit", command=on_closing  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=400, y=300)

window.mainloop()
