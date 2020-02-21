from __future__ import print_function
from __future__ import division
import cv2
import argparse
alpha_slider_max = 100
title_window = 'Linear Blend'

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def on_trackbar(val):
    print(val)
    ret, frame = cap.read()
    cv2.imshow(title_window, frame)

cv2.namedWindow(title_window)
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)

# Show some stuff
while(cap.isOpened()):
    on_trackbar(0)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Wait until user press some key
cap.release()
cv2.destroyAllWindows()