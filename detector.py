import cv2
import numpy as np
import time
import math
import sys
import context


class Detector:
    lower_yellow = np.array([45, 60, 70], dtype=np.uint8)
    upper_yellow = np.array([80, 255, 255], dtype=np.uint8)

    kernal = np.ones((3, 3), np.uint8)

    @staticmethod
    def zeros(l_bbox):
        return l_bbox[0] == 0 and l_bbox[1] == 0 and l_bbox[2] == 0 and l_bbox[3] == 0

    @staticmethod
    def convert_bbox(l_bbox):
        return l_bbox[0], l_bbox[1], l_bbox[0] + l_bbox[2], l_bbox[1] + l_bbox[3]

    @staticmethod
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

    @staticmethod
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    @staticmethod
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 3)
        return img

    @staticmethod
    def almost_equal(a, b, delta):
        return abs(a - b) < abs(np.average([a, b]) * delta)

    def detect_corners(self, img):
        blr = cv2.blur(img, (13, 13))

        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)

        l_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        op0 = cv2.erode(l_mask, self.kernal, iterations=1)
        fin = cv2.dilate(op0, self.kernal, iterations=5)

        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not len(contours) == 0:
            cnt = max(contours, key=cv2.contourArea)
            # for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                return l_mask, (x, y, w, h)
        return l_mask, (0, 0, 0, 0)
