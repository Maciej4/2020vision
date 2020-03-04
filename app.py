#!/usr/bin/env python
from importlib import import_module
import os
import cv2
from flask import Flask, render_template, Response
import socket
import context
from camera import Camera
from detector import Detector
from nt_interface import NTInterface
import threading
import time
import numpy as np
import fcntl
import struct


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


print("ip address: " + get_ip_address())

app = Flask(__name__)
# Camera = Camera.Camera
detector = Detector()
camera = Camera()
nt_interface = NTInterface()

last_frame = None
ret = None
t_bbox = None
diagnostic_img = None
proper_green = (26, 178, 79)


def draw_target(img):
    height, width, _ = img.shape
    size = 20
    cv2.line(img, (int(width / 2 - size), int(height / 2)), (int(width / 2 + size), int(height / 2)), proper_green, 2)
    cv2.line(img, (int(width / 2), int(height / 2 - size)), (int(width / 2), int(height / 2 + size)), proper_green, 2)
    return img


def draw_color_values(frame):
    height, width, _ = frame.shape
    color = frame[height / 2, width / 2]
    hsv_color = cv2.cvtColor(np.array([[color]]), cv2.COLOR_BGR2HSV)[0][0]
    cv2.putText(frame, 'o HSV: %d, %d, %d' % (hsv_color[0], hsv_color[1], hsv_color[2]), (width / 2, height / 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, proper_green, 2)


def detector_thread():
    global diagnostic_img, t_bbox, video_len, out
    vid_start_time = time.time()
    while context.keep_running:
        # get the job from the front of the queue
        laps = [1000*time.time()]
        frame = camera.get_frame()
        laps.append(1000*time.time())

        # out.write(frame)
        # laps.append(1000*time.time())

        diagnostic_img, t_bbox = detector.detect_corners(frame)
        laps.append(1000*time.time())

        nt_interface.put_num("tx", t_bbox[0] + t_bbox[2]/2)
        nt_interface.put_num("ty", t_bbox[1] + t_bbox[3]/2)
        cv2.putText(diagnostic_img, "cap:{0:.2f}ms, is:{1:.2f}ms, tot: {2:.2f}"
                    .format((laps[1]-laps[0]), (laps[2]-laps[1]), (laps[2]-laps[0])),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)


thread = threading.Thread(target=detector_thread)
context.threads.append(thread)
# this ensures the thread will die when the main thread dies
# can set t.daemon to False if you want it to keep running
thread.daemon = False
thread.start()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/diagnostic')
def diagnostic():
    """Video streaming home page."""
    return render_template('diagnostic.html')


@app.route('/color')
def color():
    """Video streaming home page."""
    return render_template('color.html')


@app.route('/both')
def both():
    """Video streaming home page."""
    return render_template('both.html')


def gen(camera):
    """Video streaming generator function."""
    while context.keep_running:
        frame = camera.get_frame()
        # draw_target(frame)

        yield encode_and_embed(frame)


def encode_and_embed(frame):
    encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')


def gen_overlay(camera):
    """Video streaming generator function."""
    while context.keep_running:
        frame = camera.get_frame()
        global t_bbox
        if len(t_bbox) == 2:
            cv2.putText(frame, 'X: %s' % round(t_bbox[0] + t_bbox[2], 3), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, 'Y: %s' % round(t_bbox[1] + t_bbox[3], 3), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        yield encode_and_embed(frame)


def diagnostic_gen(camera):
    """Video streaming generator function."""
    global diagnostic_img

    while context.keep_running:
        yield encode_and_embed(diagnostic_img)


def color_gen(camera):
    """Video streaming generator function."""
    while context.keep_running:
        frame = camera.get_frame()
        draw_color_values(frame)
        yield encode_and_embed(frame)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/diagnostic_feed')
def diagnostic_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(diagnostic_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/color_feed')
def color_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(color_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

context.keep_running = False

for thread in context.threads:
    thread.join()
