import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from pynput import keyboard
from textblob import TextBlob

# ================================
# LOAD MODELS
# ================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
model = load_model("emotion_model.hdf5", compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Surprise', 'Sad', 'Neutral']

emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Neutral': (255, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Stress': (0, 140, 255)  # Orange
}

# ================================
# TYPING TRACKERS
# ================================
typed_text = ""
key_times = []
backspace_count = 0
last_key_time = None
pauses = []

print("\nStart typing ANYWHERE. Press ESC to stop typing.\n")

def on_key_press(key):
    global typed_text, key_times, last_key_time, pauses, backspace_count

    t = time.time()
    key_times.append(t)

    # detect pauses
    if last_key_time is not None:
        if t - last_key_time > 2:
            pauses.append(t - last_key_time)
    last_key_time = t

    try:
        if hasattr(key, "char") and key.char is not None:
            typed_text += key.char
        elif key == keyboard.Key.space:
            typed_text += " "
        elif key == keyboard.Key.backspace:
            backspace_count += 1
            if len(typed_text) > 0:
                typed_text = typed_text[:-1]
        elif key == keyboard.Key.esc:
            return False
    except:
        pass

listener = keyboard.Listener(on_key_press=on_key_press)
listener.start()

# ================================
# TYPING STRESS ANALYSIS
# ================================
def analyze_typing():
    if len(key_times) < 2:
        return 0, 0, 0, 0

    n_keys = len(key_times)
    total_time = key_times[-1] - key_times[0]
    typing_speed = n_keys / total_time if total_time > 0 else 0

    avg_pause = sum(pauses) / len(pauses) if pauses else 0
    backspace_rate = backspace_count / n_keys if n_keys > 0 else 0

    stress = 0

    # Slow typing = stress
    if typing_speed < 2:
        stress += 40
    elif typing_speed < 3:
        stress += 20

    # Pause stress
    if avg_pause > 3:
        stress += 30

    # Backspace frustration
    if backspace_rate > 0.05:
        stress += 30

    return typing_speed, avg_pause, backspace_rate, min(100, stress)

# ================================
# SENTIMENT ANALYSIS
# ================================
def analyze_sentiment(text):
    if not text.strip():
        return 0.0
    polarity = TextBlob(text).sentiment.polarity  # -1 to 1
    return polarity

# ================================
# START CAMERA
# ================================
cap = cv2.VideoCapture(0)

print("Camera running... Press Q to quit camera window.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_stress_score = 0

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))

        preds = model.predict(roi, verbose=0)[0]

        max_idx = np.argmax(preds)
        top_emotion = emotion_labels[max_idx]
        confidence = preds[max_idx]

        # Face stress detection
        neg_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']
        stress_score = sum([preds[emotion_labels.index(e)] for e in neg_emotions]) / len(neg_emotions)

        if stress_score > 0.35:
            top_emotion = "Stress"
            confidence = stress_score

        face_stress_score = round(stress_score * 100, 1)

        color = emotion_colors.get(top_emotion, (255,255,255))
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 3)

        label = f"{top_emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Emotion + Typing + Stress Detection", frame)

    # ==================
    # Live Typing Metrics
    # ==================
    typing_speed, avg_pause, backspace_rate, typing_stress = analyze_typing()
    sentiment = analyze_sentiment(typed_text)

    print(f"Face Stress: {face_stress_score}% | TypingSpeed: {typing_speed:.2f} | "
          f"Pause: {avg_pause:.2f}s | BackspaceRate: {backspace_rate*100:.1f}% | "
          f"Sentiment: {sentiment:.2f} | TypingStress: {typing_stress}%     ", end="\r")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop everything
cap.release()
cv2.destroyAllWindows()
listener.stop()

print("\n\n==============================")
print(" FINAL TYPING + EXPRESSION METRICS ")
print("==============================")
print(f"Typed Text       : {typed_text}")
print(f"Sentiment Score  : {sentiment}")
print(f"Typing Speed     : {typing_speed:.2f} keys/sec")
print(f"Avg Pause        : {avg_pause:.2f} sec")
print(f"Backspace Rate   : {backspace_rate*100:.1f}%")
print(f"Typing Stress    : {typing_stress}%")
print(f"Face Stress      : {face_stress_score}%")
print("==============================\n")