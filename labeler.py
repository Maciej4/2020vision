import cv2
import sys
import numpy as np
import datetime
import csv

print("Please enter input file:")
directory = input()
cap = cv2.VideoCapture(str(directory))
cap.set(3, 640)
cap.set(4, 480)

# file_name = str(datetime.datetime.now().strftime("./videos/%y_%m_%d_%H_%M_%S.mp4"))
# print("Writing to ", file_name)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter(file_name, fourcc, 30.0, (640, 480))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video has ", frame_count, " frames")
count = 0

# Exit if video not opened.
if not cap.isOpened():
    print("Could not open video.")
    sys.exit()

# Read first frame.
ret, frame = cap.read()
if not ret:
    print("Cannot read video frame")
    sys.exit()

with open('./vision/video1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        bbox = cv2.selectROI(frame, False)

        print("Frame: [{0}/{1}]".format(count, frame_count))

        count += 1

        writer.writerow(bbox)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# out.release()
cap.release()
cv2.destroyAllWindows()
