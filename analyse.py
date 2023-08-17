import cv2
from collections import deque
from itertools import islice
import numpy as np
import math

prevposition = 0
position = 0

def getCountLineCrossed(pointList):
    global prevposition
    global position
    a = (0,225)
    b = (800,225)
    try:
        position = ((b[0] - a[0])*(pointList[0][1] - a[1]) - (b[1] - a[1])*(pointList[0][0] - a[0]))
        prevposition = ((b[0] - a[0])*(pointList[0 - 5][1] - a[1]) - (b[1] - a[1])*(pointList[0 - 5][0] - a[0]))
    except:
        pass
    
    if(prevposition != 0 and position != 0):
        if(position > 0 and prevposition < 0):
            print("crossed to right")
            return "up"
        if(position < 0 and prevposition > 0):
            print("crossed to left")
            return "down"
    return None

    

def getSpeed(pointList):
    dx = 0
    dy = 0
    try:
        for x in range(len(pointList)-1):
            dx += pointList[x+1][0] - pointList[x][0]  
            dy += pointList[x+1][1] - pointList[x][1] 

        speed = math.sqrt(abs(dx * dx - dy * dy))

        return round(speed / 100)
    except:
        pass
