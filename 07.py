import cv2
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from textblob import TextBlob
import threading
import tensorflow as tf
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model 

# -----------------------------#
# LOAD MODELS
# -----------------------------#
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

emotion_colors = {
    'Angry': (0,0,255),
    'Disgust': (0,128,0),
    'Fear': (128,0,128),
    'Happy': (0,255,0),
    'Neutral': (255,255,0),
    'Sad': (255,0,0),
    'Surprise': (0,255,255),
    'Stress': (0,140,255)
}

current_emotion = "None"
current_confidence = 0.0
stress_from_face = 0.0

# -----------------------------#
# TKINTER UI
# -----------------------------#
root = tk.Tk()
root.title("Stress & Emotion Dashboard")
root.geometry("1200x700")

# CAMERA STATS
emotion_label = tk.Label(root, text="Emotion: ---", font=("Arial", 16))
emotion_label.pack()

stress_label = tk.Label(root, text="Stress (Face): ---", font=("Arial", 16))
stress_label.pack()

# TEXTBOX FOR JOURNALING
journal_box = scrolledtext.ScrolledText(root, width=70, height=15, font=("Arial", 13))
journal_box.pack(pady=10)

typing_label = tk.Label(root, text="Typing Speed: 0 chars/sec", font=("Arial", 14))
typing_label.pack()

sentiment_label = tk.Label(root, text="Sentiment: ---", font=("Arial", 14))
sentiment_label.pack()

final_stress_label = tk.Label(root, text="FINAL Burnout Score: ---", font=("Arial", 20), fg="red")
final_stress_label.pack(pady=10)

# -----------------------------#
# TYPING BEHAVIOR VARIABLES
# -----------------------------#
keystrokes = []
backspaces = 0
pauses = []
last_key_time = None

def on_key(event):
    global last_key_time, backspaces
    
    now = time.time()
    keystrokes.append(now)

    # Detect pauses
    if last_key_time:
        delta = now - last_key_time
        if delta > 2:
            pauses.append(delta)

    last_key_time = now

    # Count backspace frustration
    if event.keysym == "BackSpace":
        backspaces += 1

journal_box.bind("<Key>", on_key)

# -----------------------------#
# TEXT + TYPING STRESS ANALYSIS
# -----------------------------#
def analyze_text():
    global stress_from_face
    
    text = journal_box.get("1.0", tk.END).strip()
    if not text:
        return

    # Typing speed
    if len(keystrokes) > 1:
        duration = keystrokes[-1] - keystrokes[0]
        speed = len(text) / duration
    else:
        speed = 0

    typing_label.config(text=f"Typing Speed: {speed:.2f} chars/sec")

    # Sentiment score
    sentiment = TextBlob(text).sentiment.polarity
    sentiment_label.config(text=f"Sentiment: {sentiment:.2f}")

    # Typing pauses
    avg_pause = np.mean(pauses) if pauses else 0

    # Backspace rate
    backspace_rate = backspaces / len(keystrokes) if keystrokes else 0

    # ------------------------------
    # FINAL BURNOUT SCORE FUSION
    # ------------------------------
    burnout_score = 0

    # Face emotion
    burnout_score += stress_from_face * 0.5

    # Sentiment effect
    if sentiment < -0.3:
        burnout_score += 20
    if sentiment < -0.6:
        burnout_score += 30

    # Typing slow
    if speed < 1.5:
        burnout_score += 20

    # High pauses
    if avg_pause > 3:
        burnout_score += 15

    # High backspace frustration
    if backspace_rate > 0.08:
        burnout_score += 20

    burnout_score = min(100, burnout_score)

    final_stress_label.config(text=f"FINAL Burnout Score: {burnout_score:.1f}%")

# -----------------------------#
# CAMERA LOOP THREAD
# -----------------------------#
def camera_loop():
    global current_emotion, stress_from_face

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48,48))
            roi = roi.astype("float32")/255.0
            roi = np.reshape(roi, (1,48,48,1))

            preds = emotion_model.predict(roi, verbose=0)[0]
            max_idx = np.argmax(preds)
            emotion = emotion_labels[max_idx]

            # stress score calculation
            neg_list = ['Angry','Disgust','Fear','Sad']
            stress_score = sum([preds[emotion_labels.index(e)] for e in neg_list])/4

            if stress_score > 0.35:
                emotion = "Stress"

            # update GUI factors
            current_emotion = emotion
            stress_from_face = stress_score * 100

            emotion_label.config(text=f"Emotion: {emotion}")
            stress_label.config(text=f"Stress (Face): {stress_from_face:.1f}%")

            color = emotion_colors.get(emotion, (255,255,255))
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

# run analysis every 1 second
def auto_update():
    analyze_text()
    root.after(1000, auto_update)

auto_update()
root.mainloop()