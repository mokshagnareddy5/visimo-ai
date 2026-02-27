// ================================
// EmotionAI — Frontend Logic
// ================================

const EMOTIONS = [
    { name: 'Angry', emoji: '😠', color: '#FF4444' },
    { name: 'Disgust', emoji: '🤢', color: '#00B400' },
    { name: 'Fear', emoji: '😨', color: '#3278C8' },
    { name: 'Happy', emoji: '😊', color: '#00E678' },
    { name: 'Sad', emoji: '😢', color: '#FF6432' },
    { name: 'Surprise', emoji: '😮', color: '#00DCFF' },
    { name: 'Neutral', emoji: '😐', color: '#E6C832' }
];

let webcam = null;
let overlay = null;
let ctx = null;
let animFrame = null;
let predicting = false;
let paused = false;
let predictInterval = null;

// ===== PAGE NAVIGATION =====
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');

    if (pageId === 'detect') {
        // Show consent first
        document.getElementById('consent-overlay').style.display = 'flex';
        document.getElementById('detect-ui').style.display = 'none';
    }

    if (pageId !== 'detect') {
        stopCamera();
    }
}

// ===== CONSENT =====
function acceptConsent() {
    document.getElementById('consent-overlay').style.display = 'none';
    document.getElementById('detect-ui').style.display = 'block';
    startCamera();
}

// ===== CAMERA =====
async function startCamera() {
    webcam = document.getElementById('webcam');
    overlay = document.getElementById('overlay');
    ctx = overlay.getContext('2d');

    // Detect mobile
    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

    try {
        const constraints = {
            video: {
                facingMode: 'user',  // Front camera (selfie)
                width: { ideal: isMobile ? 480 : 640 },
                height: { ideal: isMobile ? 360 : 480 }
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcam.srcObject = stream;

        webcam.onloadedmetadata = () => {
            overlay.width = webcam.videoWidth;
            overlay.height = webcam.videoHeight;
            initEmotionBars();
            startPrediction();
        };
    } catch (err) {
        alert('Camera access denied. Please allow camera access to use emotion detection.');
        showPage('landing');
    }
}

function stopCamera() {
    if (predictInterval) clearInterval(predictInterval);
    if (webcam && webcam.srcObject) {
        webcam.srcObject.getTracks().forEach(t => t.stop());
    }
    predicting = false;
}

function toggleCamera() {
    const btn = document.getElementById('btn-toggle');
    paused = !paused;
    btn.textContent = paused ? '▶️ Resume' : '⏸️ Pause';
}

// ===== EMOTION BARS =====
function initEmotionBars() {
    const container = document.getElementById('emotion-bars');
    container.innerHTML = EMOTIONS.map(e => `
        <div class="emotion-bar-item" id="bar-${e.name}">
            <div class="emotion-bar-header">
                <span class="emotion-name">
                    <span class="emotion-emoji">${e.emoji}</span>
                    ${e.name}
                </span>
                <span class="emotion-pct" id="pct-${e.name}">0%</span>
            </div>
            <div class="emotion-bar-bg">
                <div class="emotion-bar-fill" id="fill-${e.name}"
                     style="width:0%; background:${e.color}"></div>
            </div>
        </div>
    `).join('');
}

function updateEmotionBars(emotions) {
    if (!emotions) return;

    EMOTIONS.forEach(e => {
        const val = emotions[e.name] || 0;
        const pct = document.getElementById(`pct-${e.name}`);
        const fill = document.getElementById(`fill-${e.name}`);
        if (pct) pct.textContent = `${Math.round(val)}%`;
        if (fill) fill.style.width = `${Math.min(val, 100)}%`;
    });
}

// ===== OVERLAY DRAWING =====
function drawFaceBox(box, color) {
    if (!ctx || !box) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Scale face box to canvas
    const scaleX = overlay.width / box.frame_w;
    const scaleY = overlay.height / box.frame_h;
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const w = box.w * scaleX;
    const h = box.h * scaleY;
    const pad = 12;
    const corner = 25;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    // Corner lines
    const corners = [
        [[x - pad, y - pad], [x - pad + corner, y - pad], [x - pad, y - pad + corner]],
        [[x + w + pad, y - pad], [x + w + pad - corner, y - pad], [x + w + pad, y - pad + corner]],
        [[x - pad, y + h + pad], [x - pad + corner, y + h + pad], [x - pad, y + h + pad - corner]],
        [[x + w + pad, y + h + pad], [x + w + pad - corner, y + h + pad], [x + w + pad, y + h + pad - corner]]
    ];

    corners.forEach(c => {
        ctx.beginPath();
        ctx.moveTo(c[1][0], c[1][1]);
        ctx.lineTo(c[0][0], c[0][1]);
        ctx.lineTo(c[2][0], c[2][1]);
        ctx.stroke();
    });
}

// ===== PREDICTION =====
function startPrediction() {
    predicting = true;

    predictInterval = setInterval(async () => {
        if (paused || !predicting) return;

        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const tempCtx = canvas.getContext('2d');
        tempCtx.drawImage(webcam, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.7);

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            const data = await res.json();

            if (data.success && data.face_detected) {
                updateEmotionBars(data.emotions);
                drawFaceBox(data.face_box, data.top_color);

                // Dominant emotion
                const dom = document.getElementById('dominant-text');
                dom.textContent = `${data.top_emotion} (${data.top_confidence}%)`;
                dom.style.color = data.top_color;

                // Quality
                const qs = document.getElementById('quality-score');
                qs.textContent = data.quality.score;
                qs.className = 'status-value ' + data.quality.score.toLowerCase();

                // Confidence
                const conf = document.getElementById('confidence-display');
                conf.textContent = `${data.top_confidence}%`;
                conf.className = 'status-value ' + (data.top_confidence > 50 ? 'good' : data.top_confidence > 30 ? 'fair' : 'poor');

                // Face status
                document.getElementById('face-status').textContent = 'Detected ✅';
                document.getElementById('face-status').className = 'status-value good';

                // Warnings
                const wList = document.getElementById('warnings-list');
                if (data.quality.warnings.length > 0) {
                    wList.innerHTML = data.quality.warnings.map(w =>
                        `<div class="warning-item">⚠️ ${w}</div>`
                    ).join('');
                } else {
                    wList.innerHTML = '<span class="no-warning">✅ No warnings</span>';
                }
            } else if (data.success && !data.face_detected) {
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                document.getElementById('face-status').textContent = 'No face';
                document.getElementById('face-status').className = 'status-value poor';
                document.getElementById('dominant-text').textContent = 'No face detected';
                document.getElementById('dominant-text').style.color = '#64748b';
            }
        } catch (err) {
            console.error('Prediction error:', err);
        }
    }, 800);  // 800ms = supports ~100 concurrent users
}

// ===== PARTICLES =====
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.animationDuration = (10 + Math.random() * 20) + 's';
        p.style.animationDelay = Math.random() * 10 + 's';
        p.style.width = (2 + Math.random() * 3) + 'px';
        p.style.height = p.style.width;
        container.appendChild(p);
    }
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
});
