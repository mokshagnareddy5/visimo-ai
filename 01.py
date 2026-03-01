"""
combined_stress_detector.py
Privacy-first on-device stress-risk detector combining:
 - Typing / keystroke timing features (global keyboard listener, timing-only, no characters)
 - Facial behaviour features via MediaPipe FaceMesh (no saving frames)
 - Aggregation, buffer wiping, optional DP noise, and a demo heuristic risk score

WARNING / CONSENT:
 - This program captures keyboard events globally and accesses your webcam.
 - Only run if you consent to those sensors. Do NOT run on someone else's machine without explicit consent.
"""

import time
import threading
import math
from collections import deque
from pynput import keyboard
import numpy as np
import cv2
import mediapipe as mp
import sys
import signal

# ----------------------------
# Configuration
# ----------------------------
AGGREGATION_INTERVAL = 20.0   # seconds between aggregated feature computations
MAX_EVENTS_BUFFER = 2000      # max keystroke events to keep in memory before forced drop
CAMERA_FRAME_WIDTH = 320
CAMERA_FRAME_HEIGHT = 240
EAR_BLINK_THRESHOLD = 0.18    # heuristic threshold for blink detection (tweak)
DP_EPSILON = None             # set to a float (e.g., 0.5) to enable Laplace DP on transmitted vectors

# ----------------------------
# Global buffers (ephemeral)
# ----------------------------
# Keystroke events: list of dicts { 'type':'down'|'up', 't':float, 'backspace':bool }
_keystroke_events = deque(maxlen=MAX_EVENTS_BUFFER)
_keystroke_lock = threading.Lock()

# Facial rolling buffers
_ear_left_buf = deque(maxlen=200)
_ear_right_buf = deque(maxlen=200)
_mouth_buf = deque(maxlen=200)
_headpose_buf = deque(maxlen=200)
_blink_count = 0
_blink_lock = threading.Lock()

# Control flags
_running = True

# ----------------------------
# Utility functions
# ----------------------------
def secure_wipe_deque(dq):
    """Overwrite and clear deque"""
    try:
        for i in range(len(dq)):
            dq[i] = None
    except Exception:
        pass
    dq.clear()

def euclidean(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

def clip_vector(v, l2_bound=10.0):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= l2_bound:
        return v
    return (v / norm) * l2_bound

def laplace_noise_vector(shape, scale):
    return np.random.laplace(loc=0.0, scale=scale, size=shape)

def add_dp_noise(vec, epsilon=0.5, sensitivity=1.0):
    if epsilon is None:
        return np.array(vec)
    scale = sensitivity / epsilon
    return np.array(vec) + laplace_noise_vector(np.array(vec).shape, scale)

# ----------------------------
# Typing (keystroke) listener
# ----------------------------
def on_press(key):
    # record only timing and whether it's a backspace - DO NOT record characters
    try:
        is_backspace = (key == keyboard.Key.backspace)
    except Exception:
        is_backspace = False
    evt = {'type': 'down', 't': time.perf_counter(), 'backspace': is_backspace}
    with _keystroke_lock:
        _keystroke_events.append(evt)

def on_release(key):
    # record key up timing; avoid storing the key info other than backspace flag where feasible
    try:
        is_backspace = (key == keyboard.Key.backspace)
    except Exception:
        is_backspace = False
    evt = {'type': 'up', 't': time.perf_counter(), 'backspace': is_backspace}
    with _keystroke_lock:
        _keystroke_events.append(evt)
    # Optionally stop on escape key (not required)
    # if key == keyboard.Key.esc:
    #     return False

def typing_listener_thread():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()  # runs until program exit

# ----------------------------
# Face processing using MediaPipe FaceMesh
# ----------------------------
mp_face = mp.solutions.face_mesh

def eye_aspect_ratio(landmarks, idxs):
    # idxs: list of 6 indices for eye landmarks; landmarks are (x,y) tuples in pixels
    try:
        p = [landmarks[i] for i in idxs]
        A = euclidean(p[1], p[5])
        B = euclidean(p[2], p[4])
        C = euclidean(p[0], p[3])
        if C == 0:
            return 0.0
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception:
        return 0.0

def mouth_opening(landmarks, top_idx, bottom_idx):
    try:
        return euclidean(landmarks[top_idx], landmarks[bottom_idx])
    except Exception:
        return 0.0

def estimate_headpose_proxy(landmarks, img_size):
    # Very rough proxy: vector from forehead-ish point to nose tip normalized by diag
    try:
        w, h = img_size
        # MediaPipe indices: use 1 (nose tip) and 10-ish for forehead if present; fallback to 0
        nose = np.array(landmarks[1])
        fore = np.array(landmarks[10]) if len(landmarks) > 10 else np.array(landmarks[0])
        vec = nose - fore
        diag = math.hypot(w, h) + 1e-6
        return (vec / diag).tolist()
    except Exception:
        return [0.0, 0.0]

def camera_thread():
    global _blink_count
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith('win') else 0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Camera thread exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        last_blink_time = 0.0
        session_start = time.time()
        while _running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            small = cv2.resize(frame, (CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w, _ = small.shape
                landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

                # left/right eye approximate indices from MediaPipe
                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]

                ear_l = eye_aspect_ratio(landmarks, left_eye_idx)
                ear_r = eye_aspect_ratio(landmarks, right_eye_idx)
                ear_avg = (ear_l + ear_r) / 2.0

                # mouth open approx
                mouth_open = mouth_opening(landmarks, 13, 14)

                # head pose proxy
                head_vec = estimate_headpose_proxy(landmarks, (w,h))

                # Append to rolling buffers
                _ear_left_buf.append(ear_l)
                _ear_right_buf.append(ear_r)
                _mouth_buf.append(mouth_open)
                _headpose_buf.append(head_vec)

                # Blink detection: EAR drop under threshold then back over -> count
                now = time.time()
                if ear_avg < EAR_BLINK_THRESHOLD:
                    # debounce by time
                    if now - last_blink_time > 0.18:
                        with _blink_lock:
                            _blink_count += 1
                        last_blink_time = now

            # Do NOT save frames; just loop. Optionally sleep a tiny bit to reduce CPU
            # small processing delay
            time.sleep(0.01)

    cap.release()

# ----------------------------
# Aggregation: compute typing features from events
# ----------------------------
def compute_typing_features_and_wipe():
    """
    From _keystroke_events produce:
      - mean_hold_ms, std_hold_ms
      - mean_interkey_ms, std_interkey_ms
      - backspace_rate_per_100_keys
      - keys_pressed_count
    Then wipe raw events.
    """
    with _keystroke_lock:
        events = list(_keystroke_events)  # shallow copy
        secure_wipe_deque(_keystroke_events)  # wipe global buffer immediately

    if not events:
        return {
            'session_duration_ms': 0.0,
            'mean_hold_ms': 0.0,
            'std_hold_ms': 0.0,
            'mean_interkey_ms': 0.0,
            'std_interkey_ms': 0.0,
            'backspace_rate_per_100_keys': 0.0,
            'keys_pressed': 0
        }

    # Pair down events into hold times
    hold_times = []
    down_stack = []
    backspace_count = 0
    for ev in events:
        if ev['type'] == 'down':
            down_stack.append({'t': ev['t'], 'backspace': ev['backspace'], 'matched': False})
            if ev['backspace']:
                backspace_count += 1
        elif ev['type'] == 'up':
            # match last unmatched down
            matched = None
            for d in reversed(down_stack):
                if not d['matched']:
                    matched = d
                    break
            if matched:
                matched['matched'] = True
                matched_up = ev['t']
                hold = (matched_up - matched['t']) * 1000.0
                hold_times.append(hold)

    # Inter-key latencies using down events in chronological order
    downs = [e for e in events if e['type'] == 'down']
    downs_sorted = sorted(downs, key=lambda x: x['t'])
    inter_key = []
    for i in range(1, len(downs_sorted)):
        inter = (downs_sorted[i]['t'] - downs_sorted[i-1]['t']) * 1000.0
        inter_key.append(inter)

    keys_pressed = len(downs_sorted)
    char_estimate = max(keys_pressed - backspace_count, 1)

    def mean_std(arr):
        if not arr:
            return 0.0, 0.0
        arr = np.array(arr)
        return float(np.mean(arr)), float(np.std(arr))

    mh, sh = mean_std(hold_times)
    mi, si = mean_std(inter_key)
    backspace_rate = (backspace_count / char_estimate) * 100.0 if char_estimate > 0 else 0.0
    session_duration_ms = (events[-1]['t'] - events[0]['t']) * 1000.0 if len(events) > 1 else 0.0

    # Wipe local arrays
    hold_times = None
    downs = None
    downs_sorted = None
    inter_key = None
    down_stack = None
    events = None

    return {
        'session_duration_ms': session_duration_ms,
        'mean_hold_ms': mh,
        'std_hold_ms': sh,
        'mean_interkey_ms': mi,
        'std_interkey_ms': si,
        'backspace_rate_per_100_keys': backspace_rate,
        'keys_pressed': keys_pressed
    }

# ----------------------------
# Aggregation: read face buffers and wipe
# ----------------------------
def compute_face_features_and_wipe():
    global _blink_count
    # Copy and wipe
    with _blink_lock:
        blink_count = _blink_count
        _blink_count = 0

    ear_l = list(_ear_left_buf)
    ear_r = list(_ear_right_buf)
    mouth = list(_mouth_buf)
    headpose = list(_headpose_buf)

    secure_wipe_deque(_ear_left_buf)
    secure_wipe_deque(_ear_right_buf)
    secure_wipe_deque(_mouth_buf)
    secure_wipe_deque(_headpose_buf)

    def safe_stats(lst):
        if not lst:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        arr = np.array(lst)
        if arr.ndim == 1:
            return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                    'min': float(np.min(arr)), 'max': float(np.max(arr))}
        else:
            # list of vectors -> compute mean/std per component
            arr = np.array(arr)
            mean = np.mean(arr, axis=0).tolist()
            std = np.std(arr, axis=0).tolist()
            return {'mean': mean, 'std': std}

    features = {
        'blink_count': int(blink_count),
        'ear_left': safe_stats(ear_l),
        'ear_right': safe_stats(ear_r),
        'mouth_open': safe_stats(mouth),
        'headpose': safe_stats(headpose),
        'frames_sampled': max(len(ear_l), len(ear_r), len(mouth), len(headpose))
    }
    return features

# ----------------------------
# Simple heuristic local inference (demo)
# ----------------------------
def compute_risk_score(typing_feats, face_feats):
    """
    Example heuristic:
      - Slow mean_interkey and high backspace => typing stress
      - High blink_count per minute and low EAR => fatigue/strain
      - Larger mouth opening variance may indicate speaking/emotional expression
    This is only a demo. Replace with an on-device ML model for production.
    """
    # Typing signals
    interkey = typing_feats.get('mean_interkey_ms', 0.0)
    backspace_rate = typing_feats.get('backspace_rate_per_100_keys', 0.0)
    typing_score = 0.0
    typing_score += max(0.0, (interkey - 150.0) / 400.0)  # slower than ~150ms adds score
    typing_score += max(0.0, backspace_rate / 200.0)     # high backspace adds score

    # Face signals
    blink_rate = 0.0
    frames = face_feats.get('frames_sampled', 0)
    if frames > 0:
        # convert blink_count per AGGREGATION_INTERVAL to per minute
        blink_count = face_feats.get('blink_count', 0)
        blink_rate = (blink_count / AGGREGATION_INTERVAL) * 60.0
    ear_mean = (face_feats['ear_left']['mean'] + face_feats['ear_right']['mean']) / 2.0 if frames>0 else 0.5
    mouth_var = face_feats['mouth_open']['std'] if frames>0 else 0.0

    face_score = 0.0
    face_score += max(0.0, (blink_rate - 12.0) / 30.0)   # very high blink rate adds
    face_score += max(0.0, (0.2 - ear_mean) * 2.0)       # very low EAR suggests blinkiness/tired
    face_score += max(0.0, mouth_var / 40.0)

    # Combine
    raw = typing_score * 0.6 + face_score * 0.4
    # Normalize to 0-1 via tanh
    score = float(math.tanh(raw))
    # Provide short explanation
    reasons = []
    if typing_score > 0.15:
        reasons.append('Typing slowed / corrections increased')
    if face_score > 0.15:
        reasons.append('Facial signs (blinks/eye closure or mouth variability) detected')
    if not reasons:
        reasons = ['No strong short-term signals detected (demo)']

    return {'risk_score': score, 'reasons': reasons, 'typing_score': typing_score, 'face_score': face_score}

# ----------------------------
# Main loop: periodic aggregation & inference
# ----------------------------
def aggregator_loop():
    try:
        while _running:
            time.sleep(AGGREGATION_INTERVAL)
            typing_feats = compute_typing_features_and_wipe()
            face_feats = compute_face_features_and_wipe()

            # Optionally prepare a combined feature vector and add DP noise if configured
            vector = [
                typing_feats['mean_interkey_ms'],
                typing_feats['mean_hold_ms'],
                typing_feats['backspace_rate_per_100_keys'],
                typing_feats['keys_pressed'],
                face_feats['blink_count'],
                face_feats['ear_left']['mean'],
                face_feats['ear_right']['mean'],
                face_feats['mouth_open']['mean'] if isinstance(face_feats['mouth_open']['mean'], float) else 0.0
            ]
            # Clip & DP (if epsilon provided)
            clipped = clip_vector(vector, l2_bound=50.0)
            dp_vec = add_dp_noise(clipped, epsilon=DP_EPSILON, sensitivity=1.0)

            # Local inference (demo)
            result = compute_risk_score(typing_feats, face_feats)

            # OUTPUT: printed to console — in production feed this to local UI only
            print("="*40)
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("Typing features (local-only):", typing_feats)
            print("Face features (local-only):", face_feats)
            print("Combined clipped vector (for optional upload):", clipped.tolist())
            if DP_EPSILON is not None:
                print(f"DP-noised vector (epsilon={DP_EPSILON}):", dp_vec.tolist())
            print("Local inference (demo):", result)
            print("="*40)

    except KeyboardInterrupt:
        pass

# ----------------------------
# Graceful shutdown
# ----------------------------
def shutdown(signum=None, frame=None):
    global _running
    print("Shutting down...")
    _running = False
    # wipe buffers
    with _keystroke_lock:
        secure_wipe_deque(_keystroke_events)
    secure_wipe_deque(_ear_left_buf); secure_wipe_deque(_ear_right_buf)
    secure_wipe_deque(_mouth_buf); secure_wipe_deque(_headpose_buf)
    # allow threads to finish
    time.sleep(0.5)
    sys.exit(0)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    print("Starting combined stress detector (typing + face).")
    print("Consent reminder: this program accesses your webcam and listens to keyboard events (timing-only).")
    print("Press Ctrl+C to stop.")

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start keyboard listener in daemon thread
    t_typing = threading.Thread(target=typing_listener_thread, daemon=True)
    t_typing.start()

    # Start camera thread
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    # Start aggregator loop (main thread)
    aggregator_loop()
