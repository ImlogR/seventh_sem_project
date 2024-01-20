import cv2
import imutils
from flask import Flask, render_template, Response
import torch
# import pandas as pd 
import threading   # Library for threading -- which allows code to run in backend
import playsound   # Library for alarm sound
import smtplib     # Library for email sending
import os
from dotenv import load_dotenv
load_dotenv()

# Initializing the Flask app
app = Flask(__name__)

fire_cascade = cv2.CascadeClassifier('fire_detection.xml') # To access xml file which includes positive and negative images of fire. (Trained images)

# vid = cv2.VideoCapture(0) # To start camera this command is used "0" for laptop inbuilt camera and "1" for USB attahed camera

def play_alarm_sound_function(): # defined function to play alarm post fire detection using threading
    playsound.playsound('fire_Alarm.mp3',True) # to play alarm 
    print("Fire alarm end") # to print in console

def send_mail_function(): # defined function to send mail post fire detection using threading
    
    # recipientmail = recipientmail.lower() # To lower case mail
    recipientmail = "nikitarai0000@gmail.com" # recipients mail
    
    try:
        sender_mail = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        if not sender_mail or not sender_password:
            print("Error: Sender email or password not provided in environment variables.")
            return
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_mail, sender_password) # Senders mail ID and password
        server.sendmail('add recipients mail', recipientmail, "Warning fire accident has been reported") # recipients mail with mail message
        print("Alert mail sent sucesfully to {}".format(recipientmail)) # to print in consol to whome mail is sent
        server.close() ## To close server
        
    except Exception as e:
        print(e) # To print error if any

# Load the YOLOv5 model into the system
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Initialize the webcam 
camera = cv2.VideoCapture(0)

# Function to generate video frames through webcam
def generate_frames():
    # initial code
    runOnce = False # created boolean
    while True:
        Alarm_Status = False
        ret, frame2 = camera.read() # Value in ret is True # To read video frame
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) # To convert frame into gray color
        fire = fire_cascade.detectMultiScale(frame2, 1.2, 5) # to provide frame resolution

        ## to highlight fire with square 
        for (x,y,w,h) in fire:
            cv2.rectangle(frame2,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame2[y:y+h, x:x+w]
            print(1)

            print("Fire alarm initiated")
            threading.Thread(target=play_alarm_sound_function).start()  # To call alarm thread

            if runOnce == False:
                print("Mail send initiated")
                threading.Thread(target=send_mail_function).start() # To call alarm thread
                runOnce = True
            if runOnce == True:
                print("Mail is already sent once")
                runOnce = True

        cv2.imshow('frame', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #initial code
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
