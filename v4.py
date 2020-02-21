import cv2
import sys
import numpy as np
import time
import datetime
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

file_name = str(datetime.datetime.now().strftime("./videos/%H:%M:%S-%d-%m-%y.mp4"))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(file_name, fourcc, 1.0, (640, 480))

count = 1

lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

kernal = np.ones((5, 5), np.uint8)

cv2.namedWindow('frame')


def edit_l0(val):
    lower_yellow[0] = val


def edit_l1(val):
    lower_yellow[1] = val


def edit_l2(val):
    lower_yellow[2] = val


def edit_u0(val):
    upper_yellow[0] = val


def edit_u1(val):
    upper_yellow[1] = val


def edit_u2(val):
    upper_yellow[2] = val


cv2.createTrackbar('lower_hue', 'frame', lower_yellow[0], 255, edit_l0)
cv2.createTrackbar('upper_hue', 'frame', upper_yellow[0], 255, edit_u0)
cv2.createTrackbar('lower_sat', 'frame', lower_yellow[1], 255, edit_l1)
cv2.createTrackbar('upper_sat', 'frame', upper_yellow[1], 255, edit_u1)
cv2.createTrackbar('lower_val', 'frame', lower_yellow[2], 255, edit_l2)
cv2.createTrackbar('upper_val', 'frame', upper_yellow[2], 255, edit_u2)


# Exit if video not opened.
if not cap.isOpened():
    print("Could not open camera. Are you sure that the camera is plugged in?")
    sys.exit()

# Read first frame.
ret, frame = cap.read()
if not ret:
    print("Cannot read camera frame")
    sys.exit()


def detect(frame0):
    if ret:
        blr = cv2.blur(frame0, (13, 13))

        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

        l_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        op0 = cv2.erode(l_mask, kernal, iterations=3)
        fin = cv2.dilate(op0, kernal, iterations=4)

        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(contours) == 0:
            # cnt = max(contours, key=cv2.contourArea)
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame0, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return l_mask
    else:
        return None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timer = cv2.getTickCount()

    mask = detect(frame)

    if count % 30 == 0:
        out.write(frame)

    count += 1

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
