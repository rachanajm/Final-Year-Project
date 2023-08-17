# import the necessary packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import imutils
import time
import cv2
import os
from sort import *
from collections import deque
from analyse import *

cross_check = []
tracker = Sort()
memory = {}
pointsDict = {}
TrackedIDs = []
counter1 = 0

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("pedestrians.mp4")

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

# loop over frames from the video file stream
while True:
        peoplecount = 0
        totalSpeed = 0
        # read the next frame from the file
        (grabbed, frame) = vs.read()
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
    
vs.release()    
cv2.destroyAllWindows()
