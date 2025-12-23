# Fire and Object Detection Web App (Flask + OpenCV + YOLOv5)

This project is a Flask-based computer vision app that:
- Streams live webcam video in the browser
- Runs YOLOv5 object detection on frames
- Detects fire using an OpenCV cascade (`fire_detection.xml`)
- Plays an alarm sound when fire is detected
- Sends an email alert (once per run) when fire is detected

## Features
- Live webcam streaming endpoint (`/video_feed`)
- Fire detection using OpenCV CascadeClassifier
- Object detection using YOLOv5s via `torch.hub.load("ultralytics/yolov5", "yolov5s")`
- Alarm sound triggered in a separate thread
- Email alert triggered in a separate thread (sent once to avoid spam)

## Tech Stack
- Python
- Flask
- OpenCV
- PyTorch
- YOLOv5 (loaded via Torch Hub)
- dotenv for environment variables

## Project Structure
Folder structure:
.
├── predictedvideo/         # output of the processed video from YOLOv5
│ ├── output_video.mp4
├── static/assets/          # contains assets bootstrap and js files for UI
│ ├── bootstrap
│ ├── js
├── templates/              # HTML files for UI
│ ├── webcam.html
│ ├── home.html
│ ├── image.html
│ └── video.html
├── .env_demo               # example .env file
└── .gitignore
├── fire_Alarm.mp3          # alarm sound file
├── fire_detection.xml      # cascade model file used by OpenCV
├── image.jpg               # used by /image route
├── LICENSE                 # LICENSE
├── main.py                 # main Flask app 
├── README.md               # README
├── requirements.txt        # Required modules
├── video.mp4               # used by /video route


## Installation

1) Create and activate a virtual environment (optional but recommended):
    python3 virtualenv venv
# macOS/Linux
    source venv/bin/activate
# Windows
    venv\Scripts\activate

2) Install dependencies:
    pip install -r requirements.txt


Notes: 
- playsound can be OS-dependent. If you face issues, you may need OS-specific audio support.

- YOLOv5 is loaded from Torch Hub and may download weights on first run (requires internet the first time).

- Email Setup (Gmail): This program sends an alert using Gmail SMTP (smtp.gmail.com:587) and reads credentials from environment variables. Be sure to change the recipientmail = "email@example.com" to your email to view the sent email for sanity check (main.py line 27).

# To setup the email environment follow the steps

1) Create a .env file in the project root

Copy .env_demo to .env:
    cp .env_demo .env

Edit .env:
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password

Important:
- For Gmail, use an App Password (recommended) instead of your normal password.
- Do not commit .env to GitHub (it is ignored by .gitignore).

2) Update the recipient email (currently hardcoded)

In send_mail_function(), the recipient is hardcoded as:

recipientmail = "email@example.com"

Change it to your preferred recipient.

# Running the App
python app.py

The Flask server runs in debug mode:

Default URL: http://127.0.0.1:5000/

# Navigation/Routes

- /
Renders templates/webcam.html (webcam page)

- /video_feed
Streams MJPEG frames from generate_frames() for the browser

- /home
Renders templates/home.html

- /image
Loads image.jpg, runs YOLOv5 detection, opens an OpenCV window with results, and renders templates/image.html

- /video
Opens video.mp4, runs YOLOv5 detection frame-by-frame, displays an OpenCV window, and renders templates/video.html

## How Fire Detection Works

Each webcam frame is processed by fire_cascade.detectMultiScale(...)

When a fire region is detected:
- A rectangle is drawn around the detection
- Alarm sound plays in a background thread
- Email is sent in a background thread only once per app run (controlled by runOnce)

# Notes and Limitations

The webcam is opened globally using cv2.VideoCapture(0). If your camera index is different, change it.

The generator currently reads frames twice per loop (once for fire detection and once for YOLO). This may reduce performance or cause sync issues depending on hardware.

cv2.imshow(...) is used inside the streaming loop; this can conflict with running a web server in some environments.

Email sending requires valid credentials and SMTP access.

# Suggested Improvements

Read one frame per loop and reuse it for both fire detection and YOLO results

Move cv2.imshow usage out of server execution (or make it optional with a flag)

Make recipient email configurable via .env

Add basic rate limiting or cooldown for alarms/emails

## License
This project is licensed under the MIT License.