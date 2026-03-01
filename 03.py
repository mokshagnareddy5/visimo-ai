from flask import Flask, send_from_directory, Response, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

# ----------------------------
# FUSION ENGINE
# ----------------------------
def fuse_scores(typing, face, voice):
    t = max(0, min(typing, 100)) / 100
    f = max(0, min(face, 100)) / 100
    v = max(0, min(voice, 100)) / 100

    fusion = (0.35*t) + (0.35*f) + (0.30*v)
    stress = math.tanh(fusion * 2)
    return float(stress)

# ----------------------------
# API ENDPOINT FOR STRESS SCORE
# ----------------------------
@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    stress = fuse_scores(
        data.get("typing_score", 0),
        data.get("facial_score", 0),
        data.get("voice_score", 0)
    )
    return jsonify({
        "stress_score": stress,
        "privacy": "No text/audio/video stored. Only numeric features processed."
    })

# ----------------------------
# SERVE MAIN HTML + JS IN ONE FILE
# ----------------------------
@app.route('/')
def index():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multimodal Stress Detector (Single File)</title>

<style>
body { font-family: Arial; padding: 20px; }
.card { padding: 15px; border: 1px solid #ccc; border-radius: 10px; margin-bottom: 20px; width: fit-content; }
video { width: 320px; height: 240px; background: #000; border-radius: 10px; }
textarea { width: 100%; height: 120px; }
.meter { width: 300px; background: #eee; height: 20px; border-radius: 10px; overflow: hidden; }
.fill { background: #ff5e5e; height: 100%; width: 0%; transition: width 0.3s; }
</style>

<!-- MediaPipe -->
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>

</head>
<body>

<h1>Privacy-First Multimodal Stress Detector</h1>
<p>No storage. All processing is local. Only numeric features sent to backend.</p>

<div class="card">
  <h2>Facial Detection</h2>
  <video id="video" autoplay playsinline></video>
  <p>Face Score: <span id="faceScore">0</span></p>
  <p>Blink Rate: <span id="blinkRate">0</span></p>
</div>

<div class="card">
  <h2>Voice Analysis</h2>
  <button id="startVoice">Start Voice</button>
  <p>Pitch (Hz): <span id="pitchHz">0</span></p>
  <p>Energy Score: <span id="voiceScore">0</span></p>
</div>

<div class="card">
  <h2>Typing Behaviour</h2>
  <textarea id="typingBox" placeholder="Type here..."></textarea>
  <p>Typing Score: <span id="typingScore">0</span></p>
</div>

<div class="card">
  <h2>Final Stress Score</h2>
  <div class="meter"><div id="meterFill" class="fill"></div></div>
  <h3>Stress Score: <span id="finalScore">0</span>%</h3>
  <button onclick="sendToBackend()">Analyze</button>
</div>

<script>
// ---------------------------
// GLOBAL VARIABLES
// ---------------------------
let faceScore = 0;
let blinkCount = 0;
let lastBlink = 0;

let voiceScore = 0;
let pitchValue = 0;

let typingEvents = [];
let backspaceCount = 0;

// ---------------------------
// TYPING BEHAVIOR
// ---------------------------
const typingBox = document.getElementById("typingBox");

typingBox.addEventListener("keydown", (e) => {
    typingEvents.push(Date.now());
    if (e.key === "Backspace") backspaceCount++;
});

function getTypingScore() {
    if (typingEvents.length < 2) return 0;

    let deltas = [];
    for (let i = 1; i < typingEvents.length; i++) {
        deltas.push(typingEvents[i] - typingEvents[i - 1]);
    }

    const avg = deltas.reduce((a, b) => a + b) / deltas.length;
    let score = Math.min(100, (300 / avg) * 100);
    return Math.round(score - backspaceCount);
}

// ---------------------------
// FACE DETECTION (MEDIAPIPE)
// ---------------------------
const video = document.getElementById("video");

async function startCamera() {
    let stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" }
    });

    video.srcObject = stream;

    const faceMesh = new FaceMesh.FaceMesh({
        locateFile: (file) => 
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onFaceResults);

    const camera = new Camera(video, {
        onFrame: async () => {
            await faceMesh.send({ image: video });
        }
    });

    camera.start();
}

function onFaceResults(results) {
    if (!results.multiFaceLandmarks) return;

    let lm = results.multiFaceLandmarks[0];

    const p = (i) => [lm[i].x, lm[i].y];
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

    let EAR =
        (dist(p(159), p(145)) + dist(p(160), p(144))) /
        (2 * dist(p(33), p(133)));

    if (EAR < 0.18 && Date.now() - lastBlink > 250) {
        blinkCount++;
        lastBlink = Date.now();
    }

    faceScore = Math.min(100, Math.max(0, (0.25 - EAR) * 400));

    document.getElementById("faceScore").innerText = Math.round(faceScore);
    document.getElementById("blinkRate").innerText = blinkCount;
}

startCamera();

// ---------------------------
// VOICE PROCESSING
// ---------------------------
document.getElementById("startVoice").onclick = async () => {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    let ctx = new AudioContext();
    let analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;

    let src = ctx.createMediaStreamSource(stream);
    src.connect(analyser);

    let buffer = new Float32Array(analyser.fftSize);

    function loop() {
        analyser.getFloatTimeDomainData(buffer);

        let energy = Math.sqrt(buffer.reduce((a, b) => a + b * b, 0) / buffer.length);
        voiceScore = Math.min(100, energy * 5000);

        pitchValue = detectPitch(buffer, ctx.sampleRate);

        document.getElementById("voiceScore").innerText = Math.round(voiceScore);
        document.getElementById("pitchHz").innerText = pitchValue || 0;

        requestAnimationFrame(loop);
    }
    loop();
};

function detectPitch(buf, sr) {
    let SIZE = buf.length;
    let best = -1, bestCorr = 0;

    for (let offset = 8; offset < 1000; offset++) {
        let corr = 0;
        for (let i = 0; i < SIZE - offset; i++) {
            corr += buf[i] * buf[i + offset];
        }
        if (corr > bestCorr) {
            bestCorr = corr;
            best = offset;
        }
    }
    return best > 0 ? Math.round(sr / best) : 0;
}

// ---------------------------
// SEND TO BACKEND
// ---------------------------
async function sendToBackend() {
    const payload = {
        typing_score: getTypingScore(),
        facial_score: Math.round(faceScore),
        voice_score: Math.round(voiceScore)
    };

    const res = await fetch("/api/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    const score = Math.round(data.stress_score * 100);

    document.getElementById("finalScore").innerText = score;
    document.getElementById("meterFill").style.width = score + "%";
}

// live typing updates
setInterval(() => {
    document.getElementById("typingScore").innerText = getTypingScore();
}, 500);

</script>

</body>
</html>
"""
    return Response(html, mimetype="text/html")

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == '__main__':
    app.run("0.0.0.0", 8000, debug=True)
