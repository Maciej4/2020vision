import cv2
import numpy as np
import time
import math
import sys
import context


class Detector:
    lower_green = np.array([45, 60, 70], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    def __init__(self):
        self.scaleFactor = .25
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # self.objp = np.float32([(-5.936, .501, 0), (-4, 0, 0), (-5.377, -5.325, 0), (-7.313, -4.824, 0),
        #                         (5.936, .501, 0), (4, 0, 0), (5.377, -5.325, 0), (7.313, -4.824, 0)])
        self.objp = np.float32([(-50, 226, 0), (50, 226, 0), (24, 183, 0), (-24, 183, 0)])

        self.objp = self.sort_points(self.objp)

        self.axis = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

        self.rvecs = [[0], [0], [0]]
        self.tvecs = [[0], [0], [0]]

        # Load previously saved data with np.load('B.npz') as X:
        with np.load('calibration640_480.npz') as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
            context.dist = self.dist
            context.mtx = self.mtx

    @staticmethod
    def sort_func(input_val):
        return math.sqrt((input_val[0] + 10) * (input_val[0] + 10) + (input_val[1] + 10) * (input_val[1] + 10))

    @staticmethod
    def sort_points(points):
        # sort by y
        sorted = points[points[:, 1].argsort()]

        # split
        top, bottom = np.split(sorted, 2)

        # sort both by x
        top = top[top[:, 0].argsort()]
        bottom = bottom[bottom[:, 0].argsort()]

        return np.concatenate((top, bottom), axis=0)

    @staticmethod
    def dist_func(a, b):
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

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
        # TODO does it help?
        # img = white_balance(img)

        # cv2.medianBlur(img, 5, img)

        # Threshold the HSV image to get only blue colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        img = cv2.bitwise_and(img, img, mask=mask)

        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        img = cv2.bitwise_and(img, img, mask=mask)

        contours, h = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pnt_set = []
        diagnostic_img = img.copy()

        for cnt in contours:
            cv2.drawContours(diagnostic_img, [cnt], 0, (0, 255, 0), 3)
            approx = cnt

            approx = cv2.convexHull(approx)

            epsilon = 0.02 * cv2.arcLength(approx, True)
            # TODO arcLength should be sufficient to eliminate tiny contours

            if epsilon < 3:
                epsilon = 3

            approx = cv2.approxPolyDP(approx, epsilon, True)

            if len(approx) != 4:
                for pnt in approx:
                    cv2.circle(diagnostic_img, tuple(pnt[0]), 0, (255, 255, 0), thickness=5)  # Light Blue
                # print("len(approx) %d" % len(approx))
                continue

            sorted_box = self.sort_points(approx[:, 0, :])

            # Points are sorted in the order of ABDC, A being top left, D bottom left
            box_ab = self.dist_func(sorted_box[0], sorted_box[1])
            box_ad = self.dist_func(sorted_box[0], sorted_box[2])
            box_bc = self.dist_func(sorted_box[1], sorted_box[3])
            box_dc = self.dist_func(sorted_box[2], sorted_box[3])

            box_r = box_ad / (box_ab + 1)

            area = cv2.contourArea(approx)

            print("box_r: {0:.7f}, box_ab: {1:.3f}, box_dc: {2:.3f}, area: {3:.3f}".format(box_r, box_ab, box_dc, area))

            if (not (
                    0.2 < box_r < 0.65 and
                    area > 3000 and
                    # area < 30000 and
                    self.almost_equal(box_ad, box_bc, 0.3) and  # allow 30% difference for opposite sides
                    # self.almost_equal(box_ad, box_bc, 0.3)
                    box_ab > box_dc
            )):
                for pnt in approx:
                    cv2.circle(diagnostic_img, tuple(pnt[0]), 0, (255, 0, 0), thickness=5)  # Dark Blue
                continue

            for pnt in approx:
                pnt_set.append(tuple(pnt[0]))
                cv2.circle(diagnostic_img, tuple(pnt[0]), 0, (255, 0, 255), thickness=5)  # Pink

        return diagnostic_img, np.float32(pnt_set)

    def solve(self, img):
        # img = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
        diagnostic_img, corners = self.detect_corners(img)

        if len(corners) != len(self.objp):
            # print('no can do')
            return None, None, None, diagnostic_img

        # print("corners")
        # print(corners)
        corners = self.sort_points(corners)

        i = 0
        for pnt in corners:
            cv2.putText(img, "%s" % i, (pnt[0], pnt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            i += 1

        # print("objp: %s" % self.objp)
        # print("corners: %s" % corners)
        # print("mtx: %s" % self.mtx)
        # print("dist: %s" % self.dist)

        ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners, self.mtx, self.dist)
        cv2.putText(img, 'X: %s' % round(tvecs[0][0], 3), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Y: %s' % round(tvecs[1][0], 3), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Z: %s' % round(tvecs[2][0], 3), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        context.rvecs = rvecs
        context.tvecs = tvecs
        context.last_detection = time.time()
        return ret, rvecs, tvecs, img


detector = Detector()

cap = cv2.VideoCapture("./testvideo/1.mp4")
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    time.sleep(0.1)
    ret, frame = cap.read()

    if not ret:
        break

    a, b, c, output_image = detector.solve(frame)

    cv2.putText(output_image, "Target^", (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for pt in detector.objp:
        cv2.circle(output_image, (int(pt[0]+200), int(500-pt[1])), 0, (0, 0, 255), thickness=5)

    cv2.imshow('frame', output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
