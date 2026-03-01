"""
visimo.ai — Facial Emotion Detection Web App
Public access via ngrok tunnel — accessible from anywhere!

Usage: python app.py
First time: you'll be asked for your ngrok auth token (free signup at ngrok.com)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import base64
import io
import threading
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_file
import qrcode

app = Flask(__name__)

# ================================
# CONFIG
# ================================
PORT = 5000
NUM_THREADS = 64  # Increased from 16 to support more concurrent connections

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_COLORS = {
    'Angry': '#FF4444', 'Disgust': '#00B400', 'Fear': '#3278C8',
    'Happy': '#00E678', 'Sad': '#FF6432', 'Surprise': '#00DCFF', 'Neutral': '#E6C832'
}
IMG_SIZE = 64

# Will be set after ngrok tunnel opens
PUBLIC_URL = None

# ================================
# TFLITE MODEL (Thread-safe)
# ================================
# Model priority: mini_XCEPTION (best) > custom v2 > original
TFLITE_FILE = "fer2013_mini_XCEPTION.tflite"
HDF5_FILE = "fer2013_mini_XCEPTION.hdf5"

if not os.path.exists(TFLITE_FILE) and not os.path.exists(HDF5_FILE):
    TFLITE_FILE = "emotion_model_v2.tflite"
    HDF5_FILE = "emotion_model_v2.hdf5"

_thread_local = threading.local()

def _load_tflite_model():
    if os.path.exists(TFLITE_FILE):
        with open(TFLITE_FILE, 'rb') as f:
            return f.read()
    return None

_tflite_model_bytes = _load_tflite_model()

if _tflite_model_bytes:
    print(f"  Model: {TFLITE_FILE} ({len(_tflite_model_bytes)//1024} KB) [TFLite]")
    USE_TFLITE = True
else:
    print("  TFLite not found, using Keras (slower)")
    import tensorflow as tf
    _keras_model = tf.keras.models.load_model(HDF5_FILE, compile=False)
    USE_TFLITE = False

def get_interpreter():
    if not hasattr(_thread_local, 'interpreter'):
        import tensorflow as tf
        _thread_local.interpreter = tf.lite.Interpreter(
            model_content=_tflite_model_bytes,
            num_threads=1  # Prevent thread explosion with Waitress
        )
        _thread_local.interpreter.allocate_tensors()
    return _thread_local.interpreter

def get_face_cascade():
    if not hasattr(_thread_local, 'face_cascade'):
        _thread_local.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _thread_local.face_cascade

def predict_emotion(face_roi):
    roi_input = face_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
    if USE_TFLITE:
        interp = get_interpreter()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        interp.set_tensor(input_details[0]['index'], roi_input)
        interp.invoke()
        return interp.get_tensor(output_details[0]['index'])[0]
    else:
        return _keras_model.predict(roi_input, verbose=0)[0]

# Cascade is now thread-local

# ================================
# ROUTES
# ================================
def get_ngrok_url():
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
            data = json.loads(r.read().decode())
            for t in data.get('tunnels', []):
                if t['proto'] == 'https':
                    return t['public_url']
    except:
        pass
    return f"http://localhost:{PORT}"

@app.route('/')
def index():
    url = get_ngrok_url()
    return render_template('index.html', app_url=url, local_ip="", port=PORT)

@app.route('/qrcode.png')
def qr_code():
    url = get_ngrok_url()
    qr = qrcode.QRCode(version=1, box_size=10, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#6366f1", back_color="#0a0e1a")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'})

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(
            gray_enhanced, scaleFactor=1.05, minNeighbors=6, minSize=(60, 60)
        )

        if len(faces) == 0:
            return jsonify({'success': True, 'face_detected': False})

        fx, fy, fw, fh = faces[0]
        margin = int(fw * 0.15)
        x1, y1 = max(0, fx - margin), max(0, fy - margin)
        x2, y2 = min(w, fx + fw + margin), min(h, fy + fh + margin)

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return jsonify({'success': True, 'face_detected': False})

        # Quality
        brightness = float(np.mean(roi))
        blur = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        quality_warnings = []
        if brightness < 60: quality_warnings.append("Low Lighting")
        elif brightness > 200: quality_warnings.append("Overexposed")
        if blur < 30: quality_warnings.append("Blurry Image")
        if fw < 100: quality_warnings.append("Move Closer")
        quality_score = "Good" if not quality_warnings else ("Fair" if len(quality_warnings) == 1 else "Poor")

        # Predict
        roi_clahe = clahe.apply(roi)
        roi_resized = cv2.resize(roi_clahe, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        roi_smooth = cv2.GaussianBlur(roi_resized, (3, 3), 0)
        roi_norm = roi_smooth.astype("float32") / 255.0
        preds = predict_emotion(roi_norm)

        emotions = {emo: round(float(preds[i]) * 100, 1) for i, emo in enumerate(EMOTIONS)}
        top_emotion = EMOTIONS[int(np.argmax(preds))]
        top_confidence = round(float(np.max(preds)) * 100, 1)

        return jsonify({
            'success': True, 'face_detected': True,
            'emotions': emotions,
            'top_emotion': top_emotion,
            'top_confidence': top_confidence,
            'top_color': EMOTION_COLORS[top_emotion],
            'face_box': {'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh), 'frame_w': w, 'frame_h': h},
            'quality': {'score': quality_score, 'warnings': quality_warnings}
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ================================
# MAIN — ngrok + Waitress
# ================================
if __name__ == '__main__':
    from pyngrok import ngrok, conf
    from waitress import create_server

    print("\n" + "=" * 58)
    print("  🧠 visimo.ai — Public Emotion Detection Server")
    print("=" * 58)

    # Set ngrok auth token
    NGROK_TOKEN = "35VA3FkXdql2FojZQNub5Wzsr4q_2hBuszu96ZQE2ceBTgpnC"
    ngrok.set_auth_token(NGROK_TOKEN)

    # Open ngrok tunnel
    print("\n  🌐 Opening public tunnel...")
    try:
        tunnel = ngrok.connect(PORT, "http")
        PUBLIC_URL = tunnel.public_url
        # Force HTTPS
        if PUBLIC_URL.startswith("http://"):
            PUBLIC_URL = PUBLIC_URL.replace("http://", "https://", 1)

        print(f"\n  ✅ PUBLIC URL: {PUBLIC_URL}")
        print(f"  📱 Share this URL or scan the QR code on the landing page!")
        print(f"  🧵 Threads: {NUM_THREADS}")
        print(f"  ⚡ Engine: {'TFLite (fast)' if USE_TFLITE else 'Keras'}")
        print(f"  👥 Capacity: ~500 concurrent users (Server Limit)")
        print(f"\n  Anyone in the world can access: {PUBLIC_URL}")
        print("\n" + "=" * 58 + "\n")
        
        # Write to file for worker threads to read reliably
        with open('public_url.txt', 'w') as f:
            f.write(PUBLIC_URL)

    except Exception as e:
        print(f"\n  ❌ ngrok error: {e}")
        print("  Running locally instead...")
        PUBLIC_URL = f"http://localhost:{PORT}"

    # Start Waitress production server
    print(f"  Starting Waitress server on port {PORT}...")
    server = create_server(
        app, 
        host='0.0.0.0', 
        port=PORT, 
        threads=NUM_THREADS,
        connection_limit=1000,  # Max simultaneous connections
        channel_timeout=60      # Keep connections alive longer
    )
    server.run()
