# app.py (Flask backend)  — privacy-first (no DB, no logs)
from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def fuse_scores(typing, face, voice):
    # normalize 0-100 -> 0-1
    t = max(0, min(typing,100)) / 100.0
    f = max(0, min(face,100)) / 100.0
    v = max(0, min(voice,100)) / 100.0

    # mid-fusion style weighted sum - example weights (tweak in experiments)
    fusion = 0.35 * t + 0.35 * f + 0.30 * v

    # non-linear squash to 0-1
    score = math.tanh(fusion * 2.0)
    return float(score)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        typing_score = float(data.get('typing_score',0))
        facial_score = float(data.get('facial_score',0))
        voice_score = float(data.get('voice_score',0))

        stress = fuse_scores(typing_score, facial_score, voice_score)
        # Do NOT store anything here. Return result only.
        return jsonify({"stress_score": stress})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Starting privacy-first fusion backend on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
