import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

sum = 0
count = 1

lower_yellow = np.array([15, 50, 65], dtype=np.uint8)
upper_yellow = np.array([60, 255, 255], dtype=np.uint8)

kernal = np.ones((5, 5), np.uint8)

while (cap.isOpened()):
    start = time.time()

    ret, frame = cap.read()

    if ret == True:
        blr = cv2.blur(frame, (13, 13))

        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        op0 = cv2.erode(mask, kernal, iterations=1)
        fin = cv2.dilate(mask, kernal, iterations=3)

        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(contours) == 0:
            cnt = max(contours, key=cv2.contourArea)

            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('ero', op0)
        cv2.imshow('fin', fin)
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

        dt = (time.time() - start) * 1000
        sum += dt
        count += 1
        print(dt)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

print("Avg inference time: %d ms" % (sum / count))

cap.release()
cv2.destroyAllWindows()
