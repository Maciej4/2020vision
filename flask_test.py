import numpy as np
import cv2
from flask import Response
from flask import Flask
from flask import render_template

outputFrame = None
app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

@app.route("/")
def index():
    return render_template("index.html")