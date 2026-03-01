"""
aai_desktop.py
Interactive Affective AI (desktop, privacy-first)

- GUI: Tkinter
- Face: MediaPipe FaceMesh via OpenCV camera frames
- Voice: sounddevice audio capture -> RMS energy & autocorrelation pitch
- Typing: input field inside GUI; timing deltas captured for typing features (but not logged externally)
- Fusion: heuristic fusion of typing + face + voice -> stress score (0-100)
- Assistant: text replies + pyttsx3 TTS; tone adapts to stress score

Run with Python 3.9. Dependencies:
pip install mediapipe opencv-python sounddevice numpy scipy pyttsx3

Privacy: no raw audio/video/text is written to disk. All processing in memory.
"""

import threading
import time
import math
import queue
import sys
import platform
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox

# ---------------------------
# CONFIG
# ---------------------------
CAM_WIDTH = 320
CAM_HEIGHT = 240
AUDIO_BLOCK_SEC = 0.4   # audio frame length for analysis
AUDIO_SR = 16000
EAR_BLINK_THRESH = 0.18
BLINK_DEBOUNCE = 0.25   # seconds

# ---------------------------
# HELPERS
# ---------------------------
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_ear(landmarks, eye_indices):
    # expecting landmarks list of (x,y) in image coords
    try:
        p = [landmarks[i] for i in eye_indices]
        A = euclid(p[1], p[5])
        B = euclid(p[2], p[4])
        C = euclid(p[0], p[3])
        if C == 0: return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

def autocorr_pitch(signal, sr):
    # fast autocorrelation pitch estimate - returns freq Hz or 0
    if len(signal) < 2: return 0
    # remove DC
    sig = signal - np.mean(signal)
    if np.allclose(sig, 0): return 0
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]
    # find first reasonable peak after lag min_lag
    min_freq = 60.0
    max_freq = 800.0
    min_lag = int(sr / max_freq) if max_freq>0 else 1
    max_lag = int(sr / min_freq) if min_freq>0 else len(corr)-1
    if max_lag <= min_lag: return 0
    window = corr[min_lag:max_lag]
    if len(window)==0: return 0
    peak = np.argmax(window)
    lag = peak + min_lag
    if lag == 0: return 0
    pitch = sr / lag
    if pitch < min_freq or pitch > max_freq: return 0
    return int(pitch)

# ---------------------------
# GLOBAL STATE (thread-safe via locks where needed)
# ---------------------------
face_lock = threading.Lock()
face_features = {
    'ear_mean': 0.0,
    'ear_std': 0.0,
    'mouth_open_mean': 0.0,
    'blink_rate_per_min': 0.0,
    'frames': 0
}

typing_lock = threading.Lock()
typing_timestamps = deque(maxlen=500)   # ms timestamps for keydown
typing_backspaces = 0

audio_lock = threading.Lock()
audio_features = {
    'rms_mean': 0.0,
    'rms_std': 0.0,
    'pitch': 0.0,
    'activity_ratio': 0.0
}

# For blink detection debounce
_last_blink_time = 0.0
_blink_timestamps = deque()

# queues to communicate
_audio_queue = queue.Queue()
_stop_event = threading.Event()

# TTS engine
tts_engine = pyttsx3.init()
# Adjust voice rate / volume as desired:
tts_engine.setProperty('rate', 160)

# ---------------------------
# FACE THREAD (MediaPipe + OpenCV)
# ---------------------------
def face_thread():
    global _last_blink_time, _blink_timestamps
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=True, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    left_eye_idxs = [33,160,158,133,153,144]
    right_eye_idxs = [362,385,387,263,373,380]
    # mouth landmarks (approx)
    mouth_top = 13
    mouth_bottom = 14

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        # convert
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks and len(results.multi_face_landmarks)>0:
            lm = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            coords = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
            ear_l = compute_ear(coords, left_eye_idxs)
            ear_r = compute_ear(coords, right_eye_idxs)
            ear_avg = (ear_l + ear_r) / 2.0
            mouth_open = euclid(coords[mouth_top], coords[mouth_bottom]) if len(coords)>max(mouth_top,mouth_bottom) else 0.0

            now = time.time()
            # blink detection (EAR drop)
            if ear_avg < EAR_BLINK_THRESH and (now - _last_blink_time) > BLINK_DEBOUNCE:
                _last_blink_time = now
                _blink_timestamps.append(now)
                # keep last minute
                cutoff = now - 60.0
                while _blink_timestamps and _blink_timestamps[0] < cutoff:
                    _blink_timestamps.popleft()

            # update face_features (rolling small smoothing)
            with face_lock:
                prev_frames = face_features['frames']
                face_features['ear_mean'] = (face_features['ear_mean']*prev_frames + ear_avg) / (prev_frames + 1)
                # naive std update (not exact) - keep small window behavior
                # For simplicity, keep store of last 30 ears? But to limit complexity keep approximate:
                face_features['ear_std'] = 0.05  # placeholder small value to avoid zero
                face_features['mouth_open_mean'] = (face_features['mouth_open_mean']*prev_frames + mouth_open) / (prev_frames + 1)
                face_features['frames'] = prev_frames + 1
                face_features['blink_rate_per_min'] = len(_blink_timestamps)

        # sleep small to reduce CPU
        time.sleep(0.02)

    cap.release()
    face_mesh.close()

# ---------------------------
# AUDIO THREAD (sounddevice capture -> features)
# ---------------------------
def audio_callback(indata, frames, time_info, status):
    # Called in audio thread by sounddevice
    if status:
        # ignore statuses (like overflow) but could log
        pass
    # downmix to mono
    mono = np.mean(indata, axis=1).astype(np.float32)
    _audio_queue_put(mono)

def _audio_queue_put(buf):
    try:
        _audio_queue.put_nowait(buf)
    except queue.Full:
        pass

def audio_processor_thread():
    # consume _audio_queue into fixed-size windows for analysis
    buffer = np.array([], dtype=np.float32)
    window_samples = int(AUDIO_SR * AUDIO_BLOCK_SEC)
    voiced_count = 0
    total_frames = 0
    while not _stop_event.is_set():
        try:
            seg = _audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        buffer = np.concatenate([buffer, seg])
        while len(buffer) >= window_samples:
            frame = buffer[:window_samples]
            buffer = buffer[window_samples:]
            # compute RMS energy
            rms = np.sqrt(np.mean(frame**2) + 1e-12)
            pitch = autocorr_pitch(frame, AUDIO_SR)
            is_active = rms > 0.0015  # empirical threshold
            if is_active:
                voiced_count += 1
            total_frames += 1
            # rolling update of audio_features
            with audio_lock:
                prev_mean = audio_features['rms_mean']
                prev_count = (audio_features.get('_count',0))
                new_count = prev_count + 1
                audio_features['rms_mean'] = (prev_mean*prev_count + rms) / new_count
                audio_features['rms_std'] = 0.0  # keep simple
                audio_features['pitch'] = pitch if pitch>0 else audio_features['pitch']
                audio_features['_count'] = new_count
                audio_features['activity_ratio'] = voiced_count / max(1, total_frames)
    # end

# ---------------------------
# AUDIO STREAM STARTER
# ---------------------------
def start_audio_stream():
    # use sounddevice default input device
    try:
        stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=audio_callback)
        stream.start()
        return stream
    except Exception as e:
        print("Audio start error:", e)
        return None

# ---------------------------
# FUSION & RISK SCORING
# ---------------------------
def compute_typing_features():
    with typing_lock:
        times = list(typing_timestamps)
        bs = typing_backspaces
    if len(times) < 2:
        return {'mean_interkey_ms':0.0, 'backspace_rate':0.0}
    deltas = [(times[i]-times[i-1]) for i in range(1,len(times))]
    mean_inter = sum(deltas)/len(deltas)
    backspace_rate = (bs / max(1,len(times))) * 100
    return {'mean_interkey_ms':mean_inter, 'backspace_rate':backspace_rate}

def compute_face_summary():
    with face_lock:
        return dict(face_features)  # shallow copy

def compute_audio_summary():
    with audio_lock:
        # return copy without internal counters
        out = {k:v for k,v in audio_features.items() if not k.startswith('_')}
    return out

def fusion_score():
    t = compute_typing_features()
    f = compute_face_summary()
    a = compute_audio_summary()

    # Map each modality to 0-1 stress proxy
    # Typing: slow interkey -> stress ; high backspace -> stress
    mean_inter = t['mean_interkey_ms'] if t['mean_interkey_ms']>0 else 200.0
    typing_score = min(1.0, max(0.0, (mean_inter - 120.0) / 400.0 + (t['backspace_rate']/200.0)))

    # Face: low EAR (closed eyes) => tired/stress ; high blink rate => fatigue
    ear_mean = f.get('ear_mean', 0.25)
    blink_rate = f.get('blink_rate_per_min', 0)
    face_score = min(1.0, max(0.0, (0.25 - ear_mean)*4.0 + (blink_rate/30.0)))

    # Audio: low energy + jittery pitch -> stress; activity ratio matters
    rms = a.get('rms_mean', 0.0)
    pitch = a.get('pitch', 0.0)
    activity = a.get('activity_ratio', 0.0)
    audio_score = min(1.0, max(0.0, (0.002 - rms)*200.0 + (0 if pitch==0 else abs(220-pitch)/400.0) + (0.5-activity)))

    # weights
    w_t, w_f, w_a = 0.35, 0.35, 0.30
    fused = w_t*typing_score + w_f*face_score + w_a*audio_score
    # to 0-100
    return max(0.0, min(100.0, round(fused*100)))

# ---------------------------
# ASSISTANT LOGIC
# ---------------------------
def assistant_respond(user_text, stress_value):
    """
    Generate a short adaptive response based on stress.
    - stress_value: 0-100
    """
    s = stress_value
    if s < 25:
        tone = "calm"
        responses = [
            "Thanks for sharing — sounds like you're doing okay. Anything you'd like help with?",
            "You're looking calm — how can I assist you today?"
        ]
    elif s < 50:
        tone = "gentle"
        responses = [
            "I notice a little tension. Want a quick breathing exercise?",
            "You might be a bit stressed — would you like a short grounding practice?"
        ]
    elif s < 75:
        tone = "supportive"
        responses = [
            "I detect some stress. Try a 60-second breathing break. I can guide you.",
            "You're showing noticeable signs of strain. Would it help to talk it through?"
        ]
    else:
        tone = "urgent-support"
        responses = [
            "I detect high stress. If you're comfortable, please consider contacting someone you trust or a professional.",
            "This looks like a high-stress moment. Would you like guided breathing or resources for immediate support?"
        ]
    # simple selection
    reply = responses[int(time.time()) % len(responses)]
    # speak the reply with different voice rates based on tone
    if tone == "calm":
        tts_engine.setProperty('rate', 165)
    elif tone == "gentle":
        tts_engine.setProperty('rate', 155)
    elif tone == "supportive":
        tts_engine.setProperty('rate', 145)
    else:
        tts_engine.setProperty('rate', 135)

    # TTS speak (non-blocking via thread)
    threading.Thread(target=lambda: tts_engine.say(reply) or tts_engine.runAndWait(), daemon=True).start()
    return reply

# ---------------------------
# GUI (Tkinter)
# ---------------------------
class AAIApp:
    def __init__(self, root):
        self.root = root
        root.title("Interactive AAI — Desktop (Privacy-first)")
        # UI layout
        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky='nsew')

        # Left: camera frame and live metrics
        left = ttk.Frame(frm)
        left.grid(row=0, column=0, padx=6, sticky='nw')
        self.video_panel = tk.Label(left)
        self.video_panel.pack()
        self.face_label = tk.Label(left, text="Face score: 0")
        self.face_label.pack()
        self.blink_label = tk.Label(left, text="Blink/min: 0")
        self.blink_label.pack()

        # Middle: typing and assistant chat
        mid = ttk.Frame(frm)
        mid.grid(row=0, column=1, padx=12, sticky='n')
        ttk.Label(mid, text="Chat with AAI").pack()
        self.chat_area = scrolledtext.ScrolledText(mid, width=60, height=15, state='disabled', wrap='word')
        self.chat_area.pack()
        self.entry = tk.Entry(mid, width=60)
        self.entry.pack(pady=6)
        self.entry.bind("<Return>", self.on_send)
        ttk.Button(mid, text="Send", command=self.on_send).pack()

        # Right: controls and status
        right = ttk.Frame(frm)
        right.grid(row=0, column=2, padx=6, sticky='ne')
        ttk.Label(right, text="Stress score").pack()
        self.stress_var = tk.StringVar(value="0")
        self.stress_label = tk.Label(right, textvariable=self.stress_var, font=('Helvetica', 16))
        self.stress_label.pack()
        ttk.Button(right, text="Run fusion now", command=self.run_fusion_once).pack(pady=6)
        ttk.Button(right, text="Clear session buffers", command=self.clear_buffers).pack(pady=6)

        # start video thumbnail updater
        self.update_ui_loop()

    def clear_buffers(self):
        global typing_timestamps, typing_backspaces, _blink_timestamps
        with typing_lock:
            typing_timestamps.clear()
            global typing_backspaces
            typing_backspaces = 0
        with face_lock:
            face_features['ear_mean'] = 0.0
            face_features['mouth_open_mean'] = 0.0
            face_features['frames'] = 0
            _blink_timestamps.clear()
        with audio_lock:
            audio_features['rms_mean'] = 0.0
            audio_features['pitch'] = 0.0
            audio_features['_count'] = 0
            audio_features['activity_ratio'] = 0.0
        self.chat_system_message("Local buffers cleared.")

    def chat_system_message(self, text):
        self.chat_area['state'] = 'normal'
        self.chat_area.insert(tk.END, f"[System] {text}\n")
        self.chat_area['state'] = 'disabled'
        self.chat_area.see(tk.END)

    def on_send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        # append typed message to chat
        self.chat_area['state'] = 'normal'
        self.chat_area.insert(tk.END, f"[You] {text}\n")
        self.chat_area['state'] = 'disabled'
        self.chat_area.see(tk.END)
        self.entry.delete(0, tk.END)
        # record typing timestamp (we captured keystrokes earlier inside field? we capture on <Key> below)
        # produce response based on current fusion
        s = fusion_score()
        reply = assistant_respond(text, s)
        self.chat_area['state'] = 'normal'
        self.chat_area.insert(tk.END, f"[AAI] {reply}\n")
        self.chat_area['state'] = 'disabled'
        self.chat_area.see(tk.END)
        # update stress display
        self.stress_var.set(str(s))

    def run_fusion_once(self):
        s = fusion_score()
        self.stress_var.set(str(s))
        self.chat_system_message(f"Fusion run → stress_score={s}")

    def update_ui_loop(self):
        # update labels with current features
        f = compute_face_summary()
        a = compute_audio_summary()
        typing = compute_typing_features()
        # update face labels
        self.face_label['text'] = f"Face ear_mean: {f.get('ear_mean',0):.3f}"
        self.blink_label['text'] = f"Blink/min: {f.get('blink_rate_per_min',0)}"
        # update stress label every second
        s = fusion_score()
        self.stress_var.set(str(s))
        # show small video frame if possible (grab a frame from camera using OpenCV)
        # We will attempt to capture a frame from camera to display a thumbnail
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (320,240))
                    # convert BGR->RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # convert to Tk image
                    from PIL import Image, ImageTk
                    im = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=im)
                    self.video_panel.imgtk = imgtk
                    self.video_panel.configure(image=imgtk)
                cap.release()
        except Exception:
            # ignore; no thumbnail available
            pass

        self.root.after(1000, self.update_ui_loop)  # run again

# ---------------------------
# KEYBOARD TYPING HOOK INSIDE THE ENTRY (we capture key events on the entry widget)
# ---------------------------
def attach_typing_capture(root, entry_widget):
    def on_key(event):
        with typing_lock:
            typing_timestamps.append(int(time.time()*1000))
        # backspace handled in on_send via counting key events if desired
    entry_widget.bind("<Key>", on_key)

# ---------------------------
# STARTUP / TEARDOWN
# ---------------------------
def main():
    # start face processing thread
    ft = threading.Thread(target=face_thread, daemon=True)
    ft.start()

    # start audio input & processor thread
    ap = threading.Thread(target=audio_processor_thread, daemon=True)
    ap.start()
    # start sounddevice input stream (callback pushes to queue)
    try:
        stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=audio_callback)
        stream.start()
    except Exception as e:
        print("Warning: could not start microphone input. Audio features disabled.", e)

    # start GUI
    root = tk.Tk()
    app = AAIApp(root)
    # attach typing capture to entry
    attach_typing_capture(root, app.entry)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    # signal threads to stop
    _stop_event.set()
    try:
        stream.stop(); stream.close()
    except Exception:
        pass

if __name__ == "__main__":
    if sys.version_info < (3,9):
        print("Warning: recommend running Python 3.9 for MediaPipe compatibility.")
    main()
