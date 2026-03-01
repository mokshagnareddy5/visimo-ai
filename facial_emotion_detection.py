import cv2
import numpy as np
import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import threading
import time
from collections import deque

# ================================
# LOAD MODELS
# ================================
# Try the new model first, fall back to original
MODEL_FILE = "emotion_model_v2.hdf5"
if not os.path.exists(MODEL_FILE):
    MODEL_FILE = "emotion_model.hdf5"
    print(f"Using original model: {MODEL_FILE}")
    IMG_SIZE = 64
else:
    print(f"Using improved model: {MODEL_FILE}")
    IMG_SIZE = 48

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
model = load_model(MODEL_FILE, compile=False)
print(f"Model input shape: {model.input_shape}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

display_names = {
    'Angry': 'Anger', 'Disgust': 'Disgust', 'Fear': 'Fear',
    'Happy': 'Happiness', 'Surprise': 'Surprise',
    'Sad': 'Sadness', 'Neutral': 'Neutral',
}

emotion_colors = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 180, 0),
    'Fear':     (200, 120, 50),
    'Happy':    (0, 230, 120),
    'Neutral':  (230, 200, 50),
    'Sad':      (255, 100, 50),
    'Surprise': (0, 220, 255),
}

# ================================
# MEDIAPIPE FACE MESH
# ================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.4, min_tracking_confidence=0.4
)

mesh_line_spec = mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=0)
contour_line_spec = mp_drawing.DrawingSpec(color=(200, 230, 255), thickness=1, circle_radius=0)
dot_spec = mp_drawing.DrawingSpec(color=(180, 220, 255), thickness=1, circle_radius=1)

# ================================
# THREADED CAMERA
# ================================
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame.copy() if self.ret else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

# ================================
# TEMPORAL BUFFER
# ================================
BUFFER_SIZE = 8
prediction_buffer = deque(maxlen=BUFFER_SIZE)
MIN_FRAMES_FOR_RESULT = 2

# ================================
# QUALITY ASSESSMENT
# ================================
class QualityAssessor:
    def __init__(self):
        self.warnings = []
        self.quality_score = "Good"

    def assess(self, gray_roi, face_w, face_h, mesh_landmarks, frame_h, frame_w):
        self.warnings = []
        scores = []

        brightness = np.mean(gray_roi)
        if brightness < 60:
            self.warnings.append("Low Lighting")
            scores.append(0.3)
        elif brightness > 200:
            self.warnings.append("Overexposed")
            scores.append(0.4)
        else:
            scores.append(1.0)

        if face_w < 100:
            self.warnings.append("Too Far Away")
            scores.append(0.3)
        elif face_w < 150:
            self.warnings.append("Move Closer")
            scores.append(0.6)
        else:
            scores.append(1.0)

        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if laplacian_var < 30:
            self.warnings.append("Blurry Image")
            scores.append(0.3)
        elif laplacian_var < 60:
            self.warnings.append("Slightly Blurry")
            scores.append(0.6)
        else:
            scores.append(1.0)

        if mesh_landmarks is not None:
            visible = sum(1 for lm in mesh_landmarks.landmark if lm.visibility > 0.5)
            total = len(mesh_landmarks.landmark)
            if visible / total < 0.6:
                self.warnings.append("Face Partially Blocked")
                scores.append(0.3)
            else:
                scores.append(1.0)

        avg = np.mean(scores) if scores else 0.5
        self.quality_score = "Good" if avg >= 0.8 else "Fair" if avg >= 0.5 else "Poor"
        return avg

quality = QualityAssessor()

# ================================
# SMART CLASSIFIER
# ================================
CONFIDENCE_THRESHOLD = 0.35
MIXED_EMOTION_GAP = 0.10
MAX_DISPLAY_CONFIDENCE = 0.85
NEUTRAL_MIN_THRESHOLD = 0.45

def classify_emotion(averaged_preds):
    sorted_idx = np.argsort(averaged_preds)[::-1]
    top_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    top_conf = averaged_preds[top_idx]
    second_conf = averaged_preds[second_idx]
    top_emotion = emotion_labels[top_idx]

    entropy = -np.sum(averaged_preds * np.log(averaged_preds + 1e-10))
    max_entropy = -np.log(1.0 / len(emotion_labels))
    if entropy / max_entropy > 0.92:
        # Still show top emotion but mark as low confidence
        return display_names.get(top_emotion, top_emotion) + " (?)", top_conf, "low_conf"

    if top_conf < CONFIDENCE_THRESHOLD:
        # Show top emotion with low confidence indicator instead of "Uncertain"
        return display_names.get(top_emotion, top_emotion) + " (?)", top_conf, "low_conf"

    if top_emotion == "Neutral" and top_conf < NEUTRAL_MIN_THRESHOLD:
        if second_conf > 0.20:
            top_emotion = emotion_labels[second_idx]
            top_conf = second_conf
            return display_names.get(top_emotion, top_emotion), min(top_conf, MAX_DISPLAY_CONFIDENCE), "single"

    if (top_conf - second_conf) < MIXED_EMOTION_GAP and second_conf > 0.20:
        n1 = display_names.get(emotion_labels[top_idx], emotion_labels[top_idx])
        n2 = display_names.get(emotion_labels[second_idx], emotion_labels[second_idx])
        return f"{n1} + {n2}", min((top_conf + second_conf) / 2, MAX_DISPLAY_CONFIDENCE), "mixed"

    return display_names.get(top_emotion, top_emotion), min(top_conf, MAX_DISPLAY_CONFIDENCE), "single"

# ================================
# DRAWING HELPERS
# ================================
def draw_corner_box(frame, x, y, w, h, color, thickness=2, corner_len=25):
    x2, y2 = x + w, y + h
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
    cv2.line(frame, (x2, y), (x2 - corner_len, y), color, thickness)
    cv2.line(frame, (x2, y), (x2, y + corner_len), color, thickness)
    cv2.line(frame, (x, y2), (x + corner_len, y2), color, thickness)
    cv2.line(frame, (x, y2), (x, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

def draw_face_mesh(frame, face_landmarks):
    mp_drawing.draw_landmarks(
        image=frame, landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None, connection_drawing_spec=mesh_line_spec
    )
    mp_drawing.draw_landmarks(
        image=frame, landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=dot_spec, connection_drawing_spec=contour_line_spec
    )

def draw_sidebar(frame, predictions, emotion_labels):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_width = 250
    cv2.rectangle(overlay, (0, 0), (panel_width, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Emotion Analysis", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (15, 50), (panel_width - 15, 50), (100, 100, 100), 1)

    sorted_idx = np.argsort(predictions)[::-1]
    start_y, spacing = 85, 52

    for i, idx in enumerate(sorted_idx):
        emotion = emotion_labels[idx]
        pct = min(predictions[idx], MAX_DISPLAY_CONFIDENCE) * 100
        d_name = display_names.get(emotion, emotion)
        color = emotion_colors.get(emotion, (200, 200, 200))
        y_pos = start_y + i * spacing
        if y_pos > h - 80:
            break

        if pct < 15:
            color = (100, 100, 100)

        cx, cy, r = 35, y_pos, 18
        cv2.circle(frame, (cx, cy), r, color, 2, cv2.LINE_AA)
        if pct > 0:
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(pct * 3.6), color, 3, cv2.LINE_AA)

        pt = f"{int(pct)}%"
        ts = cv2.getTextSize(pt, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0]
        cv2.putText(frame, pt, (cx - ts[0]//2, cy + ts[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, d_name, (62, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        bx, bw, bh = 160, 70, 7
        by = cy - bh // 2
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        fw = int(bw * pct / 100)
        if fw > 0:
            cv2.rectangle(frame, (bx, by), (bx + fw, by + bh), color, -1)

def draw_quality_panel(frame, qa, reliability, trend):
    h, w = frame.shape[:2]
    pw, px = 220, w - 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "System Status", (px+12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.line(frame, (px+12, 50), (w-12, 50), (100,100,100), 1)
    y = 80

    qc = (0,230,120) if qa.quality_score=="Good" else (0,200,255) if qa.quality_score=="Fair" else (0,80,255)
    cv2.putText(frame, "Quality:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, qa.quality_score, (px+100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, qc, 2, cv2.LINE_AA)
    y += 30

    rc = (0,230,120) if reliability=="High" else (0,200,255) if reliability=="Medium" else (0,80,255)
    cv2.putText(frame, "Reliability:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, reliability, (px+115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 2, cv2.LINE_AA)
    y += 30

    tc = (0,230,120) if trend=="Stable" else (0,200,255)
    cv2.putText(frame, "Trend:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, trend, (px+85, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 2, cv2.LINE_AA)
    y += 40

    if qa.warnings:
        cv2.putText(frame, "Warnings:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,140,255), 1, cv2.LINE_AA)
        y += 25
        for wt in qa.warnings:
            cv2.putText(frame, f"! {wt}", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,255), 1, cv2.LINE_AA)
            y += 22
    else:
        cv2.putText(frame, "No warnings", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,100), 1, cv2.LINE_AA)

def draw_live_banner(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (250, 0), (w-220, 28), (20, 20, 60), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    if int(time.time()*2) % 2 == 0:
        cv2.circle(frame, (268, 14), 5, (0,0,255), -1, cv2.LINE_AA)
    cv2.putText(frame, "LIVE  -  Emotion Analysis Active  |  No Data Stored",
                (282, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,255), 1, cv2.LINE_AA)

def draw_disclaimer(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-25), (w, h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "Note: Predictions are estimates only. Accuracy may vary across demographics and conditions.",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1, cv2.LINE_AA)

# ================================
# CONSENT SCREEN
# ================================
def show_consent_screen():
    img = np.zeros((500, 800, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    cv2.putText(img, "Facial Emotion Detection System", (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.line(img, (120, 75), (680, 75), (100,100,100), 1)
    lines = [
        "PRIVACY NOTICE:", "",
        "This system uses your camera to analyze facial expressions",
        "in real-time and estimate emotional states.", "",
        "  - No images or video are saved or stored",
        "  - No data is transmitted to any server",
        "  - All processing happens locally on this device",
        "  - Predictions are estimates only, not definitive", "",
        "Press Y to accept, N to decline."
    ]
    y = 120
    for line in lines:
        color = (0,180,255) if "PRIVACY" in line else (200,200,200)
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 25
    cv2.rectangle(img, (200, 420), (350, 465), (0,180,0), -1)
    cv2.putText(img, "Y - Accept", (215, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (450, 420), (600, 465), (0,0,180), -1)
    cv2.putText(img, "N - Decline", (462, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Consent Required", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('y'), ord('Y')):
            cv2.destroyWindow("Consent Required")
            return True
        elif key in (ord('n'), ord('N')):
            cv2.destroyAllWindows()
            return False

# ================================
# HELPERS
# ================================
def get_trend(buf):
    if len(buf) < 4:
        return "Analyzing..."
    recent = [np.argmax(p) for p in list(buf)[-8:]]
    return "Stable" if len(set(recent)) <= 2 else "Changing"

def get_reliability(q, c, ct):
    if ct in ("uncertain", "insufficient"):
        return "Low"
    v = q * 0.4 + c * 0.6
    return "High" if v >= 0.50 else "Medium" if v >= 0.30 else "Low"

# ================================
# MAIN
# ================================
print("\n========================================")
print("  Real-Time Facial Emotion Detection")
print("  With Vulnerability Safeguards")
print("========================================\n")

if not show_consent_screen():
    print("User declined. Exiting.")
    exit()

print("Consent granted. Starting camera...\n")
cam = CameraStream(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

print(f"Camera: {int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("Press Q to quit.\n")

frame_count = 0
PREDICT_EVERY = 3
last_predictions = np.array([0.0] * len(emotion_labels))
last_faces = []
last_label = "Analyzing..."
last_confidence = 0.0
last_classify_type = "uncertain"
last_quality_val = 0.5
current_mesh = None

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh
    mesh_results = face_mesh.process(rgb)
    current_mesh = None
    if mesh_results.multi_face_landmarks:
        for fl in mesh_results.multi_face_landmarks:
            draw_face_mesh(frame, fl)
            current_mesh = fl

    # Emotion Detection
    frame_count += 1
    if frame_count % PREDICT_EVERY == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(
            gray_enhanced, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60)
        )
        last_faces = list(faces) if len(faces) > 0 else []

        for (fx, fy, fw, fh) in faces:
            margin = int(fw * 0.15)
            x1 = max(0, fx - margin)
            y1 = max(0, fy - margin)
            x2 = min(w, fx + fw + margin)
            y2 = min(h, fy + fh + margin)
            if x2-x1 < 20 or y2-y1 < 20:
                continue

            try:
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                last_quality_val = quality.assess(roi, fw, fh, current_mesh, h, w)

                roi_clahe = clahe.apply(roi)
                roi_resized = cv2.resize(roi_clahe, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                roi_smooth = cv2.GaussianBlur(roi_resized, (3, 3), 0)
                roi_norm = roi_smooth.astype("float32") / 255.0
                roi_input = np.reshape(roi_norm, (1, IMG_SIZE, IMG_SIZE, 1))

                preds = model.predict(roi_input, verbose=0)[0]
                prediction_buffer.append(preds)

                if len(prediction_buffer) >= MIN_FRAMES_FOR_RESULT:
                    averaged = np.mean(list(prediction_buffer), axis=0)
                    last_predictions = averaged
                    last_label, last_confidence, last_classify_type = classify_emotion(averaged)
                else:
                    last_label = "Analyzing..."
                    last_confidence = 0.0
                    last_classify_type = "uncertain"
            except Exception as e:
                continue

    # Draw results
    for (fx, fy, fw, fh) in last_faces:
        # Always use the emotion's color for the box
        color = (255, 255, 255)  # default white
        if last_classify_type == "mixed":
            color = (200, 180, 255)
        for emo, name in display_names.items():
            if name in last_label:
                color = emotion_colors.get(emo, (255, 255, 255))
                break

        padding = 15
        draw_corner_box(frame, fx-padding, fy-padding, fw+2*padding, fh+2*padding, color, 2, 30)

        label = f"{last_label} {last_confidence*100:.1f}%"
        cv2.putText(frame, label, (fx-padding, fy-padding-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Panels
    draw_sidebar(frame, last_predictions, emotion_labels)
    trend = get_trend(prediction_buffer)
    reliability = get_reliability(last_quality_val, last_confidence, last_classify_type)
    draw_quality_panel(frame, quality, reliability, trend)
    draw_live_banner(frame)
    draw_disclaimer(frame)

    cv2.imshow("Real-Time Facial Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Camera stopped. No data was stored.")
