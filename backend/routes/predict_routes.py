from flask import Blueprint, request, jsonify
import joblib
import numpy as np
from datetime import datetime
from utils.shap_handler import get_top_features
from utils.gemini_client import (
    generate_explanation,
    generate_lifestyle_suggestions,
    generate_followup_plan,
    generate_prescription_summary,
)
from utils.token import decode_token
from database.mongo import predictions_collection
from bson import ObjectId

predict = Blueprint("predict", __name__)

# Load models once
preprocessor = joblib.load("models/preprocessor.joblib")
log_model = joblib.load("models/logistic_model.joblib")
rf_model = joblib.load("models/rf_model.joblib")
xgb_model = joblib.load("models/xgb_model.joblib")

# SHAP Explainer (Tree-based)
import shap
explainer = shap.TreeExplainer(rf_model)

@predict.post("/predict")
def predict_risk():
    raw = request.json or {}
    patient_name = raw.get("patientName") or raw.get("patient_name")
    features = raw.get("features") or raw
    lifestyle = raw.get("lifestyle") or {}

    data = features
    df = {key: [value] for key, value in data.items()}

    import pandas as pd
    df = pd.DataFrame(df)
    # Ensure derived/renamed columns expected by the saved preprocessor are present.
    # Some pipelines expect features like `age_group`, `bp_cat`, `chol_cat`, and a
    # misspelled `thalch` (instead of `thalach`). Create them from raw inputs when
    # possible so transform doesn't fail with missing columns.
    if "thalach" in df.columns and "thalch" not in df.columns:
        df["thalch"] = df["thalach"]

    # age_group from `age` (simple binning that matches typical pipelines)
    if "age" in df.columns and "age_group" not in df.columns:
        def _age_group(a):
            try:
                a = float(a)
            except Exception:
                return "unknown"
            if a < 31:
                return "0-30"
            if a < 46:
                return "31-45"
            if a < 61:
                return "46-60"
            return "61+"

        df["age_group"] = df["age"].apply(_age_group)

    # bp_cat from `trestbps`
    if "trestbps" in df.columns and "bp_cat" not in df.columns:
        def _bp_cat(b):
            try:
                b = float(b)
            except Exception:
                return "unknown"
            if b < 120:
                return "normal"
            if b < 130:
                return "elevated"
            if b < 140:
                return "hypertension1"
            return "hypertension2"

        df["bp_cat"] = df["trestbps"].apply(_bp_cat)

    # chol_cat from `chol`
    if "chol" in df.columns and "chol_cat" not in df.columns:
        def _chol_cat(c):
            try:
                c = float(c)
            except Exception:
                return "unknown"
            if c < 200:
                return "normal"
            if c < 240:
                return "borderline"
            return "high"

        df["chol_cat"] = df["chol"].apply(_chol_cat)

    # Preprocess
    try:
        processed = preprocessor.transform(df)
    except ValueError as e:
        # Return a clear JSON error so frontend can show a helpful message
        return jsonify({"msg": "Preprocessing failed", "error": str(e)}), 400
    feature_names = preprocessor.get_feature_names_out()

    # Ensemble predictions
    p1 = log_model.predict_proba(processed)[0][1]
    p2 = rf_model.predict_proba(processed)[0][1]
    p3 = xgb_model.predict_proba(processed)[0][1]

    final_score = (p1 + p2 + p3) / 3

    # Risk level
    if final_score < 0.33:
        risk = "Low"
    elif final_score < 0.66:
        risk = "Moderate"
    else:
        risk = "High"

    # SHAP explanation (best-effort). If SHAP fails, return an empty list and include
    # an optional `shap_error` message in the response so the frontend can surface it.
    shap_error = None
    try:
        X_for_shap = processed.toarray() if hasattr(processed, "toarray") else processed
        top_features = get_top_features(explainer, X_for_shap, feature_names)
    except Exception as e:
        top_features = []
        shap_error = str(e)

    # Gemini-powered natural language explanation and lifestyle suggestions.
    # These calls are best-effort: if the Gemini client is not configured,
    # the helper functions fall back to simple, safe templates.
    explanation_text = generate_explanation(
        data,
        float(final_score),
        risk,
        top_features,
    )
    lifestyle_suggestions = generate_lifestyle_suggestions(
        data,
        float(final_score),
        risk,
        top_features,
    )
    followup_plan = generate_followup_plan(
        data,
        float(final_score),
        risk,
        top_features,
    )
    prescription_summary = generate_prescription_summary(
        data,
        float(final_score),
        risk,
        top_features,
    )

    # Determine user identity from JWT (if provided).
    auth_header = request.headers.get("Authorization", "")
    user_email = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = decode_token(token)
            user_email = payload.get("email")
        except Exception:
            # If token is invalid, we still allow prediction but do not tag user.
            user_email = None

    response = {
        "input": data,
        "risk_score": float(final_score),
        "risk_level": risk,
        "top_features": top_features,
        "explanation_text": explanation_text,
        "lifestyle_suggestions": lifestyle_suggestions,
        "followup_plan": followup_plan,
        "prescription_summary": prescription_summary,
        "lifestyle": lifestyle,
        "patientName": patient_name,
    }
    if shap_error:
        response["shap_error"] = shap_error

    # Persist prediction to history collection
    try:
        patient_id = None
        if user_email and patient_name:
            # Patient identity scoped per doctor
            patient_id = f"{user_email}::{patient_name.strip().lower()}"

        predictions_collection.insert_one(
            {
                "created_at": datetime.utcnow(),
                "userId": user_email,
                "doctorId": user_email,
                "patientId": patient_id,
                "patientName": patient_name,
                "input": data,
                "risk_score": float(final_score),
                "risk_level": risk,
                "trestbps": data.get("trestbps"),
                "chol": data.get("chol"),
                "thalach": data.get("thalach"),
                "oldpeak": data.get("oldpeak"),
                "restecg": data.get("restecg"),
                "smoking_status": lifestyle.get("smoking_status"),
                "diabetes_status": lifestyle.get("diabetes_status"),
                "family_history_diabetes": lifestyle.get("family_history_diabetes"),
                "pregnancy_status": lifestyle.get("pregnancy_status"),
                "top_features": top_features,
                "explanation_text": explanation_text,
                "lifestyle_suggestions": lifestyle_suggestions,
                "followup_plan": followup_plan,
                "prescription_summary": prescription_summary,
            }
        )
    except Exception:
        # Failing to write history should not break the main prediction flow
        pass

    return jsonify(response)
@predict.route("/history/<user_id>", methods=["GET", "OPTIONS"])
def get_history_for_user(user_id):
    """Return prediction history for a specific user, sorted by newest first.

    Uses JWT from the Authorization header to verify that the caller is the
    same user as the requested user_id (email).
    """
    # Let CORS preflight succeed without auth.
    if request.method == "OPTIONS":
        return ("", 200)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"msg": "Authorization header missing"}), 401

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception as e:
        return jsonify({"msg": "Invalid token", "error": str(e)}), 401

    email = payload.get("email")
    if not email or email != user_id:
        return jsonify({"msg": "Forbidden"}), 403

    try:
        cursor = (
            predictions_collection.find({"userId": user_id})
            .sort("created_at", -1)
        )
        history = []
        for doc in cursor:
            doc["id"] = str(doc.pop("_id", ""))
            created_at = doc.get("created_at")
            if hasattr(created_at, "isoformat"):
                doc["created_at"] = created_at.isoformat()
            history.append(doc)
        return jsonify({"items": history})
    except Exception as e:
        return jsonify({"msg": "Failed to load history", "error": str(e)}), 500


@predict.route("/doctor/patients", methods=["GET"])
def get_doctor_patients():
    """Return unique patients for the logged-in doctor.

    Groups predictions by patientId and patientName, returning last visit and
    assessment counts. Only accessible to users with role == "Doctor".
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"msg": "Authorization header missing"}), 401

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception as e:
        return jsonify({"msg": "Invalid token", "error": str(e)}), 401

    email = payload.get("email")
    role = payload.get("role", "Doctor")
    if not email:
        return jsonify({"msg": "Unauthorized"}), 401
    if role != "Doctor":
        return jsonify({"msg": "Forbidden"}), 403

    try:
        pipeline = [
            {"$match": {"doctorId": email, "patientId": {"$ne": None}}},
            {
                "$group": {
                    "_id": "$patientId",
                    "patientName": {"$first": "$patientName"},
                    "lastVisit": {"$max": "$created_at"},
                    "assessmentCount": {"$sum": 1},
                }
            },
            {"$sort": {"lastVisit": -1}},
        ]
        cursor = predictions_collection.aggregate(pipeline)
        patients = []
        for doc in cursor:
            last_visit = doc.get("lastVisit")
            patients.append(
                {
                    "patientId": doc.get("_id"),
                    "patientName": doc.get("patientName"),
                    "lastVisit": last_visit.isoformat() if hasattr(last_visit, "isoformat") else None,
                    "assessmentCount": doc.get("assessmentCount", 0),
                }
            )
        return jsonify({"patients": patients})
    except Exception as e:
        return jsonify({"msg": "Failed to load patients", "error": str(e)}), 500


@predict.route("/doctor/patient/<patient_id>", methods=["GET"])
def get_doctor_patient_profile(patient_id):
    """Return full prediction history for a specific patient of the doctor."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"msg": "Authorization header missing"}), 401

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception as e:
        return jsonify({"msg": "Invalid token", "error": str(e)}), 401

    email = payload.get("email")
    role = payload.get("role", "Doctor")
    if not email:
        return jsonify({"msg": "Unauthorized"}), 401
    if role != "Doctor":
        return jsonify({"msg": "Forbidden"}), 403

    try:
        cursor = (
            predictions_collection.find({"doctorId": email, "patientId": patient_id})
            .sort("created_at", -1)
        )
        history = []
        first_visit = None
        last_visit = None
        patient_name = None
        for doc in cursor:
            if patient_name is None:
                patient_name = doc.get("patientName")
            created_at = doc.get("created_at")
            if created_at is not None:
                if last_visit is None:
                    last_visit = created_at
                first_visit = created_at if first_visit is None else min(first_visit, created_at)
            item = {
                "id": str(doc.get("_id")),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
                "risk_score": doc.get("risk_score"),
                "risk_level": doc.get("risk_level"),
                "trestbps": doc.get("trestbps"),
                "chol": doc.get("chol"),
                "thalach": doc.get("thalach"),
                "oldpeak": doc.get("oldpeak"),
                "restecg": doc.get("restecg"),
                "smoking_status": doc.get("smoking_status"),
                "diabetes_status": doc.get("diabetes_status"),
                "family_history_diabetes": doc.get("family_history_diabetes"),
                "pregnancy_status": doc.get("pregnancy_status"),
                "input": doc.get("input", {}),
            }
            history.append(item)

        stats = {
            "assessmentCount": len(history),
            "firstVisit": first_visit.isoformat() if hasattr(first_visit, "isoformat") else None,
            "lastVisit": last_visit.isoformat() if hasattr(last_visit, "isoformat") else None,
        }

        return jsonify(
            {
                "patientId": patient_id,
                "patientName": patient_name,
                "stats": stats,
                "history": history,
            }
        )
    except Exception as e:
        return jsonify({"msg": "Failed to load patient profile", "error": str(e)}), 500


@predict.route("/history/item/<item_id>", methods=["DELETE", "OPTIONS"])
def delete_history_item(item_id):
    """Delete a single prediction history item for the logged-in user.

    Uses JWT to ensure the record belongs to the caller (userId == email).
    """
    if request.method == "OPTIONS":
        return ("", 200)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"msg": "Authorization header missing"}), 401

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception as e:
        return jsonify({"msg": "Invalid token", "error": str(e)}), 401

    email = payload.get("email")
    if not email:
        return jsonify({"msg": "Unauthorized"}), 401

    try:
        oid = ObjectId(item_id)
    except Exception:
        return jsonify({"msg": "Invalid id"}), 400

    try:
        res = predictions_collection.delete_one({"_id": oid, "userId": email})
        if res.deleted_count == 0:
            return jsonify({"msg": "Item not found"}), 404
        return jsonify({"msg": "Deleted"})
    except Exception as e:
        return jsonify({"msg": "Failed to delete history item", "error": str(e)}), 500
