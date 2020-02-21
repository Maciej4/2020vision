import numpy as np
import cv2
import time
import datetime
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

file_name = str(datetime.datetime.now().strftime("./videos/%H:%M:%S-%d-%m-%y.mp4"))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(file_name, fourcc, 1.0, (640, 480))

time_sum = 0
count = 1

lower_yellow = np.array([16, 210, 122], dtype=np.uint8)
upper_yellow = np.array([25, 255, 255], dtype=np.uint8)

kernal = np.ones((5, 5), np.uint8)

p_x = -1
p_y = -1
d_p = -1
pps = -1
p_tl = False
v_max = 0
p_t = 0

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

while cap.isOpened():
    start = time.time()

    ret, frame = cap.read()

    if ret:
        blr = cv2.blur(frame, (13, 13))

        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        op0 = cv2.erode(mask, kernal, iterations=3)
        fin = cv2.dilate(mask, kernal, iterations=3)

        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(contours) == 0:
            cnt = max(contours, key=cv2.contourArea)

            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if not p_x == -1:
                    cv2.rectangle(frame, (p_x-10, p_y-10), (p_x+20, p_y+20), (255, 0, 0), 2)
                    d_p = math.sqrt(math.pow((x+w/2.0)-p_x,2.0)+math.pow((y+h/2.0)-p_y,2.0))
                p_x = int(x + w/2.0)
                p_y = int(y + h/2.0)
                p_tl = True
        else:
            p_x = -1
            p_y = -1
            d_p = -1

        if count % 30 == 0:
            out.write(frame)

        cv2.imshow('ero', op0)
        cv2.imshow('fin', fin)
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

        dt = (time.time() - start) * 1000
        if not d_p == -1:
            pps = d_p / (time.time() - p_t)
            if v_max < pps:
                v_max = pps
            print("Test: dt: {0}s, d_p: {1}, pps: {2}, d_x{3}".format(dt/1000.0, d_p, pps, (x+w/2)-p_x))  # * 0.00381))
            p_t = time.time()
            # print(d_p)
            # print(pps)# * 0.00381)
        else:
            if p_tl:
                p_tl = False
                print("Target Lost")
                print("Stats: max_vel: {0}m/s".format(v_max * 0.00381))
                v_max = 0
        time_sum += dt
        count += 1
        # print(dt)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

print("Avg inference time: %d ms" % (time_sum / count))

out.release()
cap.release()
cv2.destroyAllWindows()
