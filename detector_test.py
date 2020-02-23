import cv2 as cv
import numpy as np
import time
import math
import sys
from networktables import NetworkTables
import glob
import matplotlib.pyplot as plt
from detector import Detector

ip = 'roborio-972-frc.local'
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

images = glob.glob('../images/valid/*.png')

detector = Detector()

for fname in images:
    startTime = time.time();
    #frame = cv.imread(fname, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    frame = cv.imread(fname, cv.COLOR_BGR2RGB)
    #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ret, rvecs, tvecs, diagnostic_img = detector.solve(frame)
    #print('R: %s' % rvecs)
    if ret is not None:
        print('T: %s' % tvecs)

        # TODO indicate the target has not been found
        sd.putNumber("visionX", tvecs[0][0])
        sd.putNumber("visionY", tvecs[1][0])
        sd.putNumber("visionZ", tvecs[2][0])

    cv.imshow('diagnostic', diagnostic_img)
    cv.imshow('frame', frame)

    # Output time in milliseconds of processing time
    #print("runtime ms: ", round((time.time()-startTime)*1000))

    # esc to kill program
    k = cv.waitKey() & 0xFF
    if k == 27:
        break
