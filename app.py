from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Load Keras model
from tensorflow.keras.models import load_model
model = load_model(os.path.join(BASE_DIR, "models", "model.h5"))


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body:
    {
        "variance": float,
        "skewness": float,
        "curtosis": float,
        "entropy": float
    }

    Returns:
    {
        "prediction": "Real" | "Fake",
        "confidence": float   (0–100 %)
    }
    """
    data = request.get_json(force=True)

    required_fields = ["variance", "skewness", "curtosis", "entropy"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        features = np.array([[
            float(data["variance"]),
            float(data["skewness"]),
            float(data["curtosis"]),
            float(data["entropy"]),
        ]])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # Scale & predict
    features_scaled = scaler.transform(features)
    raw = model.predict(features_scaled)          # shape (1, 1) for binary sigmoid

    prob_fake = float(raw[0][0])                  # sigmoid → probability of class 1 (Fake)
    is_fake   = prob_fake >= 0.5
    label     = "Fake" if is_fake else "Real"
    confidence = prob_fake * 100 if is_fake else (1 - prob_fake) * 100

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)