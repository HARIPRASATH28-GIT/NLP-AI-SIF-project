from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from flask_cors import CORS
from supabase import create_client
import joblib
import os
import re
import numpy as np
load_dotenv()

# =========================================================
# 1. FLASK SETUP
# =========================================================

app = Flask(__name__)
CORS(app)
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

# =========================================================
# 2. SUPABASE CONFIGURATION
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set in backend/.env"
    )
# =========================================================
# 3. LOAD MODEL FILES
# =========================================================

BASE_DIR = BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nlp_model = joblib.load(
 os.path.join(BASE_DIR, "sif_nlp_model.pkl")
)

tfidf = joblib.load(
 os.path.join(BASE_DIR, "tfidf_precursor.pkl")
)

precursor_rules = joblib.load(
 os.path.join(BASE_DIR, "precursor_rules.pkl")
)

safety_critical_combinations = joblib.load(
 os.path.join(BASE_DIR, "safety_critical_combinations.pkl")
)

risk_df = joblib.load(
 os.path.join(BASE_DIR, "risk_df.pkl")
)

model_config = joblib.load(
 os.path.join(BASE_DIR, "model_config.pkl")
)

print("✅ All SIF model files loaded")


# =========================================================
# 4. PRECURSOR DETECTION
# =========================================================

precursor_categories = [
 "Fall from elevation",
 "Struck-by hazard",
 "Caught-in / Caught-between",
 "Electrical hazard",
 "Hazardous energy / LOTO",
 "Machine guarding",
 "Inadequate PPE",
 "Hazardous positioning / human factor",
 "Explosion / Fire / Energy release",
 "Heat / Environmental exposure"
]


def detect_precursors(text):

 text = str(text).lower()

 detected = []

 rules = {
     "Fall from elevation": [
         "fell from",
         "fall from",
         "fell off",
         "fall off",
         "elevated",
         "height",
         "scaffold",
         "platform",
         "ladder",
         "roof"
     ],

     "Struck-by hazard": [
         "struck by",
         "hit by",
         "struck",
         "struck against",
         "falling object",
         "object fell",
         "vehicle hit",
         "truck hit"
     ],

     "Caught-in / Caught-between": [
         "caught between",
         "caught in",
         "caught under",
         "trapped between",
         "crushed between",
         "pinched"
     ],

     "Electrical hazard": [
         "electrical",
         "electric shock",
         "electrocut",
         "power line",
         "energized",
         "live wire",
         "voltage"
     ],

     "Hazardous energy / LOTO": [
         "lockout",
         "lock out",
         "tagout",
         "tag out",
         "loto",
         "stored energy",
         "unexpected startup"
     ],

     "Machine guarding": [
         "machine guard",
         "guarding",
         "unguarded",
         "guard removed",
         "moving machinery",
         "rotating equipment"
     ],

     "Inadequate PPE": [
         "without ppe",
         "without fall protection",
         "no ppe",
         "no helmet",
         "no gloves",
         "no harness",
         "not wearing",
         "missing ppe"
     ],

     "Hazardous positioning / human factor": [
         "wrong position",
         "improper position",
         "positioned",
         "line of fire",
         "standing under",
         "too close",
         "unsafe position"
     ],

     "Explosion / Fire / Energy release": [
         "explosion",
         "exploded",
         "fire",
         "flame",
         "blast",
         "burn",
         "ignition",
         "energy release"
     ],

     "Heat / Environmental exposure": [
         "heat stress",
         "heat exposure",
         "hot environment",
         "high temperature",
         "dehydration",
         "heat exhaustion"
     ]
 }

 for category, keywords in rules.items():

     for keyword in keywords:

         if keyword in text:

             detected.append(category)
             break

 return detected


# =========================================================
# 5. NLP PRECURSOR DETECTION
# =========================================================

nlp_category_names = [
 "Fall from elevation",
 "Struck-by hazard",
 "Caught-in / Caught-between",
 "Electrical hazard",
 "Explosion / Fire / Energy release",
 "Machine guarding"
]


def get_confident_nlp_precursors(text):

 try:

     X = tfidf.transform([text])

     decision_scores = nlp_model.decision_function(X)

     if decision_scores.ndim == 1:
         decision_scores = decision_scores.reshape(1, -1)

     scores = decision_scores[0]

     predictions = []

     confidence_results = {}

     for i, category in enumerate(nlp_category_names):

         score = float(scores[i])

         confidence = 1 / (1 + np.exp(-score))

         confidence_percent = round(
             confidence * 100,
             2
         )

         confidence_results[category] = confidence_percent

         if confidence_percent >= 60:

             predictions.append(category)

     return predictions, confidence_results

 except Exception as e:

     print("⚠️ NLP detection error:", e)

     return [], {}


# =========================================================
# 6. DATA-DRIVEN RISK SCORING
# =========================================================

def calculate_corrected_data_risk(precursors):

 if not precursors:

     return {
         "raw_score": 0,
         "evidence_score": 0,
         "priority": "LOW",
         "detected_count": 0
     }

 detected_rows = risk_df[
     risk_df["Precursor"].isin(precursors)
 ]

 if detected_rows.empty:

     return {
         "raw_score": 0,
         "evidence_score": 0,
         "priority": "LOW",
         "detected_count": len(precursors)
     }

 total_weight = detected_rows[
     "Data_Driven_Weight"
 ].sum()

 total_confidence = detected_rows[
     "Confidence"
 ].sum()

 if total_confidence > 0:

     weighted_evidence = (
         (
             detected_rows["Data_Driven_Weight"]
             * detected_rows["Confidence"]
         ).sum()
         / total_confidence
     )

 else:

     weighted_evidence = 0

 evidence_score = 50 + (
     weighted_evidence - 25
 )

 if len(precursors) >= 3:

     evidence_score += 10

 elif len(precursors) == 2:

     evidence_score += 5

 evidence_score = max(
     0,
     min(evidence_score, 100)
 )

 if evidence_score >= 75:

     priority = "CRITICAL"

 elif evidence_score >= 55:

     priority = "HIGH"

 elif evidence_score >= 35:

     priority = "MEDIUM"

 else:

     priority = "LOW"

 return {
     "raw_score": round(total_weight, 2),
     "evidence_score": round(evidence_score, 2),
     "priority": priority,
     "detected_count": len(precursors)
 }


# =========================================================
# 7. HYBRID SIF PREDICTION
# =========================================================

def hybrid_sif_prediction(incident_text):

 incident_text = str(incident_text).strip()

 if not incident_text:

     return None

 # Rule-based detection
 rule_precursors = detect_precursors(
     incident_text
 )

 # NLP detection
 nlp_precursors, confidence_results = (
     get_confident_nlp_precursors(
         incident_text
     )
 )

 # Combine both
 hybrid_precursors = list(
     dict.fromkeys(
         rule_precursors + nlp_precursors
     )
 )

 # Risk calculation
 risk_result = calculate_corrected_data_risk(
     hybrid_precursors
 )

 # Safety-critical combinations
 precursor_set = set(
     hybrid_precursors
 )

 matched_combinations = []

 for combination in safety_critical_combinations:

     if set(combination).issubset(
         precursor_set
     ):

         matched_combinations.append(
             " + ".join(
                 sorted(combination)
             )
         )

 safety_override = (
     len(matched_combinations) > 0
 )

 initial_priority = risk_result[
     "priority"
 ]

 final_priority = initial_priority

 if (
     safety_override
     and initial_priority in ["LOW", "MEDIUM"]
 ):

     final_priority = "HIGH"

 final_score = risk_result[
     "evidence_score"
 ]

 return {

     "incident": incident_text,

     "rule_precursors":
         rule_precursors,

     "nlp_precursors":
         nlp_precursors,

     "hybrid_precursors":
         hybrid_precursors,

     "nlp_confidence":
         confidence_results,

     "raw_evidence_score":
         risk_result["raw_score"],

     "risk_score":
         final_score,

     "initial_priority":
         initial_priority,

     "safety_critical":
         safety_override,

     "matched_combinations":
         matched_combinations,

     "final_priority":
         final_priority
 }


# =========================================================
# 8. SAVE RESULT TO SUPABASE
# =========================================================

def save_sif_result(result):

 record = {

     "incident_text":
         result["incident"],

     "risk_score":
         result["risk_score"],

     "priority":
         result["final_priority"],

     "safety_critical":
         result["safety_critical"],

     "rule_precursors":
         result["rule_precursors"],

     "nlp_precursors":
         result["nlp_precursors"],

     "hybrid_precursors":
         result["hybrid_precursors"],

     "matched_combinations":
         result["matched_combinations"],

     "nlp_confidence":
         result["nlp_confidence"]
 }

 response = (
     supabase
     .table("sif_incidents")
     .insert(record)
     .execute()
 )

 return response


# =========================================================
# 9. HOME API
# =========================================================




# =========================================================
# 10. PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

 try:

     data = request.get_json()

     if not data:

         return jsonify({
             "error":
                 "Request body is empty"
         }), 400

     incident = data.get(
         "incident",
         ""
     )

     if not incident.strip():

         return jsonify({
             "error":
                 "Incident description is required"
         }), 400

     # Run AI prediction
     result = hybrid_sif_prediction(
         incident
     )

     # Save to Supabase
     save_sif_result(result)

     return jsonify(result)

 except Exception as e:

     print(
         "❌ Prediction error:",
         str(e)
     )

     return jsonify({
         "error": str(e)
     }), 500


# =========================================================
# 11. RUN SERVER
# =========================================================

if __name__ == "__main__":

 print(
     "🚀 SIF Flask Backend starting..."
 )

 app.run(
     host="0.0.0.0",
     port=5000,
     debug=False
 )
