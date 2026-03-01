from flask import Flask, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

def compute_risk(typing, facial, voice):
    t = typing / 100
    f = facial / 100
    v = voice / 100

    fusion = (0.4*t + 0.35*f + 0.25*v)
    stress_score = float(math.tanh(fusion * 2))

    return stress_score

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json

    typing_score = data["typing_score"]
    facial_score = data["facial_score"]
    voice_score = data["voice_score"]

    stress_score = compute_risk(typing_score, facial_score, voice_score)

    return jsonify({
        "stress_score": stress_score,
        "message": "No data stored. Computed locally & discarded."
    })

if __name__ == '__main__':
    app.run(debug=True)
