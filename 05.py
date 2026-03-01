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
#model.save("Emotion_Model")  # Save the model in TensorFlow SavedModel format   
# Emotion labels based on FER-2013 dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Suprise', 'Sad', 'Neutral']

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
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Extract face ROI
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))

        # Predict emotion
        predictions = model.predict(roi)[0]
        max_index = np.argmax(predictions)
        emotion = emotion_labels[max_index]

        # Display label
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Emotion Detection - OpenCV", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()