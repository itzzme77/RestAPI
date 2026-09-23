# Raphael: AI Soil Nutrient Diagnostic System (Flask Web Application)

Raphael is an AI-based soil diagnostic application built with **Flask** that evaluates primary macronutrient status (**Nitrogen**, **Phosphorus**, and **Potassium**) from soil sample measurements and applies a rule-based **Chemical Constraint Layer (CCL)** to detect pH-related nutrient availability restrictions.

---

## Required Files

```text
.
├── app.py                      # Flask web application
├── raphael_mhNN.keras          # Pre-trained multi-head neural network
├── raphael_scaler.pkl          # Pre-fitted StandardScaler
├── requirements.txt            # Python package dependencies
├── templates/
│   └── index.html              # HTML5 dashboard template
└── static/
    └── css/
        └── style.css           # Modern stylesheet
```

---

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Flask web dashboard with:

```bash
python app.py
```

Then open your browser at `http://localhost:5000`.

---

## System Architecture

### 1. Multi-Head Neural Network (mhNN)
- **Input Features (4 features in exact order)**:
  - `N` — Total Nitrogen (g/kg)
  - `P` — Available Phosphorus (mg/kg)
  - `K` — Available Potassium (mg/kg)
  - `pH_H2O` — Soil pH (measured in H₂O)
- **Preprocessing**: `StandardScaler` (`raphael_scaler.pkl`) applies standard feature scaling `(x - mean) / scale`.
- **Output Heads**:
  - `N_status` — 3 classes (`0: DEFICIENT`, `1: SUFFICIENT`, `2: HIGH`)
  - `P_status` — 3 classes (`0: DEFICIENT`, `1: SUFFICIENT`, `2: HIGH`)
  - `K_status` — 3 classes (`0: DEFICIENT`, `1: SUFFICIENT`, `2: HIGH`)
- **Confidence**: Model confidence is calculated from the maximum softmax probability for each corresponding head.

### 2. Learned Classification Thresholds
- **Nitrogen ($N$)** [Total N, g/kg]: Deficient $< 1.93$ | Sufficient $1.93 - 3.02$ | High $> 3.02$
- **Phosphorus ($P$)** [Available P, mg/kg]: Deficient $< 14.8$ | Sufficient $14.8 - 29.7$ | High $> 29.7$
- **Potassium ($K$)** [Extractable K, mg/kg]: Deficient $< 99.7$ | Sufficient $99.7 - 198.6$ | High $> 198.6$

### 3. Chemical Constraint Layer (CCL)
The Chemical Constraint Layer (CCL) is a rule-based constraint system based on established soil chemistry solubility and fixation dynamics. It operates independently from the neural network predictions:

- **Nitrogen**: Restricted if `pH_H2O < 6.0` or `pH_H2O > 8.0`
- **Phosphorus**: Restricted if `pH_H2O < 6.0` or `pH_H2O > 7.5`
- **Potassium**: Restricted if `pH_H2O < 6.0` or `pH_H2O > 9.0`

> **Note**: Nutrient status is produced by the trained neural network. pH-related availability restriction is determined separately using the CCL pH rules. If a nutrient is classified as SUFFICIENT by the neural network but is restricted by pH, both facts are presented separately without overwriting the neural network status.

---

## API Usage

The application also exposes a JSON API endpoint:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"n": 2.5, "p": 22.0, "k": 150.0, "ph": 6.5}'
```
