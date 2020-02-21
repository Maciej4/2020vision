import cv2
import sys
import numpy as np
import time
import datetime
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

tracker = cv2.TrackerMedianFlow_create()

file_name = str(datetime.datetime.now().strftime("./videos/%H:%M:%S-%d-%m-%y.mp4"))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(file_name, fourcc, 1.0, (640, 480))

time_sum = 0
count = 1

lower_yellow = np.array([16, 210, 122], dtype=np.uint8)
upper_yellow = np.array([25, 255, 255], dtype=np.uint8)

kernal = np.ones((5, 5), np.uint8)

tracking = False
track_started = False

rep_dets = 0

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


def start_track(l_bbox):
    global tracker
    global tracking
    tracker = cv2.TrackerMedianFlow_create()
    l_ok = tracker.init(frame, l_bbox)
    if not l_ok:
        print("Tracking not cleanly started")
        return
    tracking = True


def track(frame0):
    global tracking
    l_ok, l_bbox = tracker.update(frame)
    if not l_ok:
        tracking = False
    cv2.putText(frame, "Tracking", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    p1 = (int(l_bbox[0]), int(l_bbox[1]))
    p2 = (int(l_bbox[0] + l_bbox[2]), int(l_bbox[1] + l_bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    return l_ok, l_bbox


def detect(frame0):
    if ret:
        blr = cv2.blur(frame0, (13, 13))

        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        op0 = cv2.erode(mask, kernal, iterations=3)
        fin = cv2.dilate(mask, kernal, iterations=3)

        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(contours) == 0:
            cnt = max(contours, key=cv2.contourArea)

            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Detecting", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                return True, (x, y, w, h)
        else:
            cv2.putText(frame, "Not Tracking", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    return False, (0, 0, 0, 0)


def get_bbox_area(l_bbox):
    return (l_bbox[0] + l_bbox[2]) * (l_bbox[1] + l_bbox[3])


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timer = cv2.getTickCount()

    if tracking:
        t_ok, t_bbox = track(frame)

        if count % 30 == 5 and t_ok:
            d_ok, d_bbox = detect(frame)
            if d_ok and math.fabs(get_bbox_area(t_bbox) - get_bbox_area(d_bbox)) > 1000:
                start_track(d_bbox)
    else:
        ok, bbox = detect(frame)
        print(rep_dets)
        if ok:
            if 10 < rep_dets:
                if track_started:
                    start_track(bbox)
                else:
                    print("Tracking started")
                    start_track(bbox)
                    track_started = True
            rep_dets += 1
        else:
            rep_dets = 0

    if count % 30 == 0:
        out.write(frame)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()