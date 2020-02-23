import cv2
import sys
import numpy as np
import time
import datetime
import math
import csv

cap = cv2.VideoCapture("./testvideo/1.mp4")
cap.set(3, 640)
cap.set(4, 480)

# file_name = str(datetime.datetime.now().strftime("./videos/%H:%M:%S-%d-%m-%y.mp4"))
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter(file_name, fourcc, 1.0, (640, 480))

count = 0

lower_yellow = np.array([45, 60, 70], dtype=np.uint8)
upper_yellow = np.array([80, 255, 255], dtype=np.uint8)

kernal = np.ones((3, 3), np.uint8)

cv2.namedWindow('frame')

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open camera. Are you sure that the camera is plugged in?")
    sys.exit()


def zeros(l_bbox):
    return l_bbox[0] == 0 and l_bbox[1] == 0 and l_bbox[2] == 0 and l_bbox[3] == 0


def convert_bbox(l_bbox):
    return l_bbox[0], l_bbox[1], l_bbox[0] + l_bbox[2], l_bbox[1] + l_bbox[3]


def intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    l_iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return l_iou


def detect(frame0):
    blr = cv2.blur(frame0, (13, 13))

    hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

    l_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    op0 = cv2.erode(l_mask, kernal, iterations=1)
    fin = cv2.dilate(op0, kernal, iterations=5)

    cv2.imshow("ero", op0)

    contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not len(contours) == 0:
        cnt = max(contours, key=cv2.contourArea)
        # for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame0, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return l_mask, (x, y, w, h)
    return l_mask, (0, 0, 0, 0)


print("Starting program!")

test_frames = 0
test_sum = 0
past_active = False
target_missed_count = 0

with open('./vision/video1.csv') as file:
    reader = csv.reader(file)
    rows = list(reader)
    csv_length = len(rows)

    if csv_length == 0:
        sys.exit()

    while cap.isOpened():
        if past_active:
            time.sleep(0.2)

        ret, frame = cap.read()
        if not ret:
            break

        if count + 1 > csv_length:
            break

        r_bbox = rows[count]

        for i in range(0, 4):
            r_bbox[i] = int(r_bbox[i])

        r_bbox = (r_bbox[0], r_bbox[1], r_bbox[2], r_bbox[3])

        mask, d_bbox = detect(frame)

        count += 1

        if zeros(r_bbox):
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)
            iou = -1.0
            # print(iou)
            past_active = False
        else:
            # Calculate stats
            iou = intersection_over_union(convert_bbox(r_bbox), convert_bbox(d_bbox))
            # print(iou)
            test_sum += iou
            test_frames += 1
            if zeros(d_bbox):
                target_missed_count += 1

            # Draw data onto frame
            cv2.rectangle(frame, (r_bbox[0], r_bbox[1]), (r_bbox[0] + r_bbox[2], r_bbox[1] + r_bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, "IOU: %.4f" % iou, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            # Show images
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)

            # Make the program wait on start of next run
            past_active = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("---Final Statistics---")
print("Key frame count: {0}".format(test_frames))
print("Avg accuracy: {0}".format(test_sum / test_frames))
print("Tgt not detected: {0} times".format(target_missed_count))

# out.release()
cap.release()
cv2.destroyAllWindows()
