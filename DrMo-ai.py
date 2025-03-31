import tkinter as tk
from tkinter import messagebox
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import mediapipe as mp
from datetime import datetime
import pytz
import pyttsx3
import math
from twilio.rest import Client
from threading import Thread
from PIL import Image, ImageTk
import warnings

# Filter out specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='cvlib')

# Twilio configuration (Replace with valid credentials before use)
account_sid = 'REPLACE_WITH_YOUR_SID'
auth_token = 'REPLACE_WITH_YOUR_AUTH_TOKEN'
client = Client(account_sid, auth_token)

def send_sms(to, body):
    message = client.messages.create(
        body=body,
        from_='REPLACE_WITH_YOUR_NUMBER',
        to=to
    )
    print(message.sid)

# Mediapipe configuration
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def get_eye_aspect_ratio(eye):
    A = math.sqrt((eye[1][0] - eye[5][0]) * 2 + (eye[1][1] - eye[5][1]) * 2)
    B = math.sqrt((eye[2][0] - eye[4][0]) * 2 + (eye[2][1] - eye[4][1]) * 2)
    C = math.sqrt((eye[0][0] - eye[3][0]) * 2 + (eye[0][1] - eye[3][1]) * 2)
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 5
counter = 0
cap = None
is_running = False

def speak(text, lang='en'):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def start_detection():
    global cap, counter, is_running
    if is_running:
        messagebox.showinfo("Information", "Detection is already running.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    counter = 0
    is_running = True
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Running", fg="green")

    def process_frame():
        global counter

        if not is_running:
            return

        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Error: Unable to access the camera", fg="red")
            return

        jordan_tz = pytz.timezone('Asia/Amman')
        jordan_time = datetime.now(jordan_tz)
        hours = jordan_time.hour
        minutes = jordan_time.minute
        seconds = jordan_time.second

        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.5, model='yolov3-tiny')
        phone_in_hand = 'cell phone' in label

        if phone_in_hand:
            speak("Don't touch the phone", 'en')
            send_sms('REPLACE_WITH_YOUR_PHONE_NUMBER', 'Warning: Phone usage detected!')
            print(f"{hours}:{minutes}:{seconds}")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in
                            [33, 160, 158, 133, 153, 144]]
                right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in
                             [362, 385, 387, 263, 373, 380]]

                left_eye = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for (x, y) in left_eye]
                right_eye = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for (x, y) in right_eye]

                left_ear = get_eye_aspect_ratio(left_eye)
                right_ear = get_eye_aspect_ratio(right_eye)

                if left_ear < EYE_AR_THRESH and right_ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        speak("Wake up", 'en')
                        send_sms('REPLACE_WITH_YOUR_PHONE_NUMBER', 'Warning: Drowsiness detected!')
                        print(f"{hours}:{minutes}:{seconds}")
                        counter = 0
                else:
                    counter = 0
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        frame = cv2.resize(frame, (1280, 720))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

        if is_running:
            video_panel.after(10, process_frame)

    process_frame()

def stop_detection():
    global cap, is_running
    if not is_running:
        messagebox.showinfo("Information", "Detection is not running.")
        return

    is_running = False
    if cap is not None:
        cap.release()
        cap = None
    video_panel.config(image='')
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Stopped", fg="red")

root = tk.Tk()
root.title("Detection System")
root.geometry("1400x900")

instructions = tk.Label(root, text="Click 'Start Detection' to begin. Click 'Stop Detection' to end.", font=("Helvetica", 14))
instructions.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

start_button = tk.Button(button_frame, text="Start Detection", command=start_detection, width=20, font=("Helvetica", 12))
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, width=20, font=("Helvetica", 12))
stop_button.pack(side=tk.RIGHT, padx=10)
stop_button.config(state=tk.DISABLED)

video_panel = tk.Label(root, width=1280, height=720)
video_panel.pack(padx=10, pady=10)

status_label = tk.Label(root, text="Status: Stopped", font=("Helvetica", 12), fg="red")
status_label.pack(pady=10)

root.mainloop()
