import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("raphael_flask")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Constants & Class Mappings
FEATURE_NAMES = ["N", "P", "K", "pH_H2O"]
CLASS_LABELS = {
    0: "DEFICIENT",
    1: "SUFFICIENT",
    2: "HIGH"
}

# Global references to model and scaler
model = None
scaler = None


def find_file(filename: str) -> Path:
    """Find a file in the application directory or in models/ subdirectory."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / filename,
        base_dir / "models" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Required file '{filename}' was not found in '{base_dir}' or '{base_dir / 'models'}'."
    )


def initialize_artifacts():
    """
    Load the pre-trained Keras multi-head neural network and StandardScaler.
    Executed once at application startup.
    No retraining or fitting is performed.
    """
    global model, scaler

    # 1. Load Scaler
    scaler_path = find_file("raphael_scaler.pkl")
    try:
        import joblib
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded StandardScaler from {scaler_path} using joblib.")
    except Exception as e1:
        try:
            import pickle
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info(f"Loaded StandardScaler from {scaler_path} using pickle.")
        except Exception as e2:
            logger.error(f"Failed to load scaler from {scaler_path}. Joblib error: {e1}; Pickle error: {e2}")
            raise RuntimeError(f"Failed to load scaler. {e1}")

    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is not None and n_features != 4:
        raise ValueError(f"StandardScaler expects {n_features} features, but exactly 4 are required.")

    # 2. Load Model
    model_path = find_file("raphael_mhNN.keras")
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Loaded Keras multi-head model from {model_path}.")
    except Exception as e:
        logger.error(f"Failed to load Keras model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model. {e}")


def evaluate_ccl(ph_val: float) -> Dict[str, Any]:
    """
    Chemical Constraint Layer (CCL):
    Rule-based constraint layer evaluating pH-related availability restrictions.
    Initial chart-derived boundaries:
      - Nitrogen: restricted if pH_H2O < 6.0 or pH_H2O > 8.0
      - Phosphorus: restricted if pH_H2O < 6.0 or pH_H2O > 7.5
      - Potassium: restricted if pH_H2O < 6.0 or pH_H2O > 9.0
    """
    restrictions = {
        "Nitrogen": ph_val < 6.0 or ph_val > 8.0,
        "Phosphorus": ph_val < 6.0 or ph_val > 7.5,
        "Potassium": ph_val < 6.0 or ph_val > 9.0,
    }
    affected = [nutrient for nutrient, is_restricted in restrictions.items() if is_restricted]
    has_restriction = len(affected) > 0
    return {
        "has_restriction": has_restriction,
        "affected_nutrients": affected,
        "restrictions": restrictions,
    }


def predict_soil_nutrients(n: float, p: float, k: float, ph: float) -> Dict[str, Any]:
    """
    Preprocess inputs in exact feature order ['N', 'P', 'K', 'pH_H2O'],
    apply StandardScaler transform, and obtain predictions from the multi-head NN.
    """
    global model, scaler
    if model is None or scaler is None:
        initialize_artifacts()

    # Exact feature order: ["N", "P", "K", "pH_H2O"]
    input_data = np.array([[float(n), float(p), float(k), float(ph)]], dtype=np.float32)

    # Apply saved StandardScaler
    scaled_data = scaler.transform(input_data)

    # Pass scaled input to trained model
    raw_outputs = model(scaled_data, training=False)

    # Unpack heads: N_status, P_status, K_status
    if isinstance(raw_outputs, list):
        n_tensor, p_tensor, k_tensor = raw_outputs[0], raw_outputs[1], raw_outputs[2]
    elif isinstance(raw_outputs, dict):
        n_tensor = raw_outputs["N_status"]
        p_tensor = raw_outputs["P_status"]
        k_tensor = raw_outputs["K_status"]
    else:
        raw_np = np.array(raw_outputs)
        n_tensor, p_tensor, k_tensor = raw_np[0], raw_np[1], raw_np[2]

    n_probs = np.array(n_tensor)[0]
    p_probs = np.array(p_tensor)[0]
    k_probs = np.array(k_tensor)[0]

    # Convert each output using argmax
    results = {}
    for name, probs in [("Nitrogen", n_probs), ("Phosphorus", p_probs), ("Potassium", k_probs)]:
        class_idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100.0)
        status_label = CLASS_LABELS[class_idx]
        results[name] = {
            "class_idx": class_idx,
            "status": status_label,
            "confidence": confidence,
            "probabilities": {
                CLASS_LABELS[0]: float(probs[0] * 100.0),
                CLASS_LABELS[1]: float(probs[1] * 100.0),
                CLASS_LABELS[2]: float(probs[2] * 100.0),
            },
        }

    return results


@app.route("/", methods=["GET"])
def index():
    """Render main dashboard with default calibrated values."""
    default_inputs = {
        "n": 2.50,
        "p": 22.00,
        "k": 150.00,
        "ph": 6.50
    }
    return render_template("index.html", inputs=default_inputs, results=None, ccl=None, errors=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle soil analysis form submission."""
    errors = []
    inputs = {}

    try:
        n_val = float(request.form.get("n", "").strip())
        inputs["n"] = n_val
        if n_val < 0.0:
            errors.append("Nitrogen (N) must be greater than or equal to 0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Nitrogen (N).")
        inputs["n"] = 2.50

    try:
        p_val = float(request.form.get("p", "").strip())
        inputs["p"] = p_val
        if p_val < 0.0:
            errors.append("Phosphorus (P) must be greater than or equal to 0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Phosphorus (P).")
        inputs["p"] = 22.00

    try:
        k_val = float(request.form.get("k", "").strip())
        inputs["k"] = k_val
        if k_val < 0.0:
            errors.append("Potassium (K) must be greater than or equal to 0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Potassium (K).")
        inputs["k"] = 150.00

    try:
        ph_val = float(request.form.get("ph", "").strip())
        inputs["ph"] = ph_val
        if not (0.0 <= ph_val <= 14.0):
            errors.append("Soil pH (H2O) must be between 0.0 and 14.0.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid numeric value for Soil pH (H2O).")
        inputs["ph"] = 6.50

    if errors:
        return render_template("index.html", inputs=inputs, results=None, ccl=None, errors=errors)

    try:
        pred_results = predict_soil_nutrients(inputs["n"], inputs["p"], inputs["k"], inputs["ph"])
        ccl_results = evaluate_ccl(inputs["ph"])
        return render_template("index.html", inputs=inputs, results=pred_results, ccl=ccl_results, errors=None)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return render_template("index.html", inputs=inputs, results=None, ccl=None, errors=[f"System Error: {e}"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic or AJAX inference."""
    data = request.get_json(silent=True) or {}
    try:
        n = float(data.get("n", 0.0))
        p = float(data.get("p", 0.0))
        k = float(data.get("k", 0.0))
        ph = float(data.get("ph", 7.0))
        
        if n < 0 or p < 0 or k < 0 or not (0.0 <= ph <= 14.0):
            return jsonify({"status": "error", "message": "Input values out of valid ranges"}), 400

        pred_results = predict_soil_nutrients(n, p, k, ph)
        ccl_results = evaluate_ccl(ph)

        return jsonify({
            "status": "success",
            "inputs": {"N": n, "P": p, "K": k, "pH_H2O": ph},
            "diagnostics": pred_results,
            "chemical_constraint_layer": ccl_results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Initialize model and scaler before starting server
    initialize_artifacts()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  Raphael AI Soil Diagnostic System (Flask)")
    print(f"  Server running at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
