"""
Interactive Affective AI (Improved Version)
-------------------------------------------
✔ Smooth camera (320x240 high-FPS)
✔ Stress descriptions (Low / Mild / High / Critical)
✔ Color-coded stress indicator
✔ Real-time facial + voice + typing fusion
✔ Desktop GUI (Tkinter)
✔ 100% Privacy — No data stored
"""

import threading
import time
import math
import sys
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import scrolledtext, ttk

# =========================
# CONFIG / CONSTANTS
# =========================
CAM_W = 320
CAM_H = 240

AUDIO_SR = 16000
AUDIO_BLOCK_SEC = 0.4

EAR_BLINK_THRESH = 0.18
BLINK_DEBOUNCE = 0.25

# =========================
# GLOBAL STATE
# =========================
# Facial
face_lock = threading.Lock()
face_data = {
    "ear": 0,
    "blink_pm": 0,
    "mouth": 0,
    "frame_count": 0
}
blink_times = deque()
last_blink = 0

# Typing
typing_lock = threading.Lock()
typing_times = deque(maxlen=300)
typing_bs = 0

# Audio
audio_lock = threading.Lock()
audio_data = {
    "rms": 0,
    "pitch": 0,
    "activity": 0,
    "_count": 0,
    "_voiced": 0
}

stop_event = threading.Event()

# TTS
tts = pyttsx3.init()
tts.setProperty("rate", 160)

# =========================
# HELPER FUNCTIONS
# =========================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def autocorr_pitch(x, sr):
    x = x - np.mean(x)
    corr = correlate(x, x, mode="full")
    corr = corr[len(corr)//2:]
    min_lag = int(sr / 800)
    max_lag = int(sr / 60)
    if max_lag <= min_lag:
        return 0
    window = corr[min_lag:max_lag]
    if len(window) == 0:
        return 0
    lag = np.argmax(window) + min_lag
    return int(sr / lag)

# =========================
# FACE THREAD
# =========================
def face_thread():
    global last_blink, blink_times

    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    left_eye = [33,160,158,133,153,144]
    right_eye = [362,385,387,263,373,380]
    mouth_top, mouth_bottom = 13, 14

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        small = cv2.resize(frame, (160, 120))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = small.shape
            coords = [(int(p.x * w), int(p.y * h)) for p in lm]

            def EAR(eye):
                a = dist(coords[eye[1]], coords[eye[5]])
                b = dist(coords[eye[2]], coords[eye[4]])
                c = dist(coords[eye[0]], coords[eye[3]])
                return (a+b)/(2*c) if c>0 else 0

            ear_val = (EAR(left_eye) + EAR(right_eye)) / 2
            mouth_open = dist(coords[mouth_top], coords[mouth_bottom])

            now = time.time()
            if ear_val < EAR_BLINK_THRESH and (now - last_blink) > BLINK_DEBOUNCE:
                blink_times.append(now)
                last_blink = now

            # prune old blinks
            while blink_times and blink_times[0] < now - 60:
                blink_times.popleft()

            with face_lock:
                face_data["ear"] = ear_val
                face_data["blink_pm"] = len(blink_times)
                face_data["mouth"] = mouth_open
                face_data["frame_count"] += 1

        time.sleep(0.01)

    cap.release()

# =========================
# AUDIO PROCESSING
# =========================
def audio_callback(indata, frames, time_info, status):
    mono = np.mean(indata, axis=1)
    process_audio_block(mono)

def process_audio_block(x):
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    pitch = autocorr_pitch(x, AUDIO_SR)
    active = 1 if rms > 0.001 else 0

    with audio_lock:
        c = audio_data["_count"] + 1
        audio_data["_count"] = c
        audio_data["rms"] = (audio_data["rms"]*(c-1) + rms)/c
        audio_data["pitch"] = pitch if pitch>0 else audio_data["pitch"]
        audio_data["_voiced"] += active
        audio_data["activity"] = audio_data["_voiced"]/c

def start_audio():
    try:
        stream = sd.InputStream(
            samplerate=AUDIO_SR, channels=1, callback=audio_callback
        )
        stream.start()
        return stream
    except Exception as e:
        print("Audio error:", e)
        return None

# =========================
# FUSION LOGIC
# =========================
def get_typing_features():
    with typing_lock:
        ts = list(typing_times)
        bs = typing_bs

    if len(ts)<2:
        return 200, 0

    deltas = [ts[i]-ts[i-1] for i in range(1,len(ts))]
    mean_inter = sum(deltas)/len(deltas)
    back_rate = (bs/len(ts))*100
    return mean_inter, back_rate

def fusion():
    mean_inter, back_rate = get_typing_features()
    face = dict(face_data)
    audio = dict(audio_data)

    # Typing score
    typing_s = min(1, max(0, (mean_inter-120)/400 + back_rate/200))

    # Face score
    ear = face.get("ear",0.3)
    blinks = face.get("blink_pm",0)
    face_s = min(1, max(0, (0.25-ear)*4 + blinks/30))

    # Audio score
    rms = audio.get("rms",0)
    pitch = audio.get("pitch",200)
    activity = audio.get("activity",0.3)
    aud_s = min(1, max(0, (0.002-rms)*200 + abs(220-pitch)/400 + (0.5-activity)))

    fused = 0.35*typing_s + 0.35*face_s + 0.30*aud_s
    return int(fused*100)

def stress_description(score):
    if score <= 25:
        return "Low Stress — Calm & stable", "green"
    elif score <= 50:
        return "Mild Stress — Slight tension", "yellow"
    elif score <= 75:
        return "High Stress — Noticeable strain", "orange"
    else:
        return "Critical Stress — Heavy load", "red"

# =========================
# AAI RESPONSE
# =========================
def generate_response(msg, score):
    if score < 25:
        r = "You're calm right now. How can I help?"
    elif score < 50:
        r = "I sense mild tension. Want a small break?"
    elif score < 75:
        r = "You're under noticeable stress. Let's try a grounding exercise."
    else:
        r = "Your stress seems high. Let's slow down. Deep breath with me?"

    threading.Thread(target=lambda: tts.say(r) or tts.runAndWait(), daemon=True).start()
    return r

# =========================
# TKINTER GUI
# =========================
class AAI:
    def __init__(self, root):
        self.root = root
        root.title("Interactive AAI (Improved)")
        frm = ttk.Frame(root, padding=10)
        frm.grid()

        # Camera panel
        self.cam_label = tk.Label(frm)
        self.cam_label.grid(row=0, column=0, rowspan=3)

        # Stress panel
        self.stress_lbl = tk.Label(frm, text="Stress: 0", font=("Arial",16))
        self.stress_lbl.grid(row=0, column=1, sticky="w")

        self.desc_lbl = tk.Label(frm, text="Status...", font=("Arial",12))
        self.desc_lbl.grid(row=1, column=1, sticky="w")

        # Chat
        self.chat = scrolledtext.ScrolledText(frm, width=50, height=12)
        self.chat.grid(row=2, column=1)
        self.chat.insert(tk.END, "[AAI] Hello! I'm here for you.\n")
        self.chat.configure(state="disabled")

        self.entry = tk.Entry(frm, width=50)
        self.entry.grid(row=3, column=1)
        self.entry.bind("<Return>", self.send)

        # typing capture
        self.entry.bind("<Key>", self.key_capture)

        self.update_gui()

    def key_capture(self, event):
        global typing_times, typing_bs
        with typing_lock:
            typing_times.append(time.time()*1000)
            if event.keysym == "BackSpace":
                typing_bs += 1

    def send(self, event=None):
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)

        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"[You] {msg}\n")
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

        score = fusion()
        reply = generate_response(msg, score)

        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"[AAI] {reply}\n")
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

    def update_gui(self):
        # Update stress info
        score = fusion()
        desc, color = stress_description(score)

        self.stress_lbl.config(text=f"Stress: {score}", fg=color)
        self.desc_lbl.config(text=desc, fg=color)

        # Update camera preview
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320,240))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(master=self.cam_label, data=cv2.imencode('.png', frame)[1].tobytes(), format="png")
            self.cam_label.imgtk = img
            self.cam_label.config(image=img)

        self.root.after(200, self.update_gui)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Start camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # Start threads
    threading.Thread(target=face_thread, daemon=True).start()
    audio_stream = start_audio()

    # Start GUI
    root = tk.Tk()
    app = AAI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    cap.release()
