import cv2
import imutils
from flask import Flask, render_template, Response
import torch
import pandas as pd 

# Initializing the Flask app
app = Flask(__name__)

# Load the YOLOv5 model into the system
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Initialize the webcam 
camera = cv2.VideoCapture(0)

# Function to generate video frames through webcam
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Performing the  object detection using YOLOv5 model
            results = model(frame)
            frame = results.render()[0]

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routing the path for the home page
@app.route('/')
def index():
    return render_template('webcam.html')

# Routing to stream the webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routing the app to the home page
@app.route("/home")
def home():
    return render_template("home.html")

# Routing the app to the image detection page
@app.route("/image")
def image():
    image = cv2.imread("image.jpg")
    image = imutils.resize(image, width=min(500, image.shape[1]))
    
    # Performing object detection using YOLOv5
    results = model(image)
    image = results.render()[0]

    cv2.imshow("Object Detection from Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render_template("image.html")

# Routing the app to the video detection page
@app.route("/video")
def video():
    cap = cv2.VideoCapture("video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Performing object detection using YOLOv5
            results = model(frame)
            frame = results.render()[0]

            cv2.imshow("Object Detection from Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            break
    return render_template("video.html")

# Runing the Flask app
if __name__ == '__main__':
    app.run(debug=True)
