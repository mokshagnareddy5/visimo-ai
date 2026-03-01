import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# ------------------------------
# Load Models
# ------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

model = load_model("emotion_model.hdf5", compile=False)

# Original FER emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Surprise', 'Sad', 'Neutral']

# NEW colors including STRESS
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 128, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Neutral': (255, 255, 0), 
    'Sad': (255, 0, 0),
    'Surprise': (0, 255, 255),
    'Stress': (0, 140, 255)     # ORANGE
}

# ------------------------------
# Start Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # ROI for prediction
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))          # FIXED
        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))     # FIXED

        preds = model.predict(roi, verbose=0)[0]

        # Extract top emotion
        max_idx = np.argmax(preds)
        top_emotion = emotion_labels[max_idx]
        confidence = preds[max_idx]

        # --------------------------
        # ADDING STRESS (custom logic)
        # --------------------------

        # negative emotions score
        neg_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']
        stress_score = sum([preds[emotion_labels.index(e)] for e in neg_emotions]) / len(neg_emotions)

        if stress_score > 0.35:   # threshold adjustable
            top_emotion = "Stress"
            confidence = stress_score

        # Draw colored box
        color = emotion_colors.get(top_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Label
        label = f"{top_emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Emotion + Stress Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()