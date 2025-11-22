import os
from typing import Any, Dict, List

import google.generativeai as genai


_API_KEY_ENV = "GEMINI_API_KEY"
_MODEL_NAME = "gemini-2.5-flash"


def _get_client() -> bool:
    """Configure Gemini client if API key is available.

    Returns True if the client is configured, False otherwise.
    """
    api_key = os.getenv(_API_KEY_ENV)
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


def _basic_explanation_fallback(risk_level: str, risk_score: float) -> str:
    return (
        f"Your predicted heart disease risk is {risk_level} "
        f"(score: {risk_score:.2f}). This tool is for educational use only and "
        f"cannot provide a medical diagnosis. Please discuss any concerns with a "
        f"qualified healthcare professional."
    )


def _basic_lifestyle_fallback(risk_level: str) -> List[str]:
    base = [
        "This tool does not give medical advice. For personalized guidance, "
        "please consult a doctor.",
    ]
    if risk_level == "Low":
        base.append(
            "Maintain a heart-healthy lifestyle with regular physical activity, "
            "balanced diet, and avoiding smoking."
        )
    elif risk_level == "Moderate":
        base.append(
            "Consider speaking with a healthcare professional about blood "
            "pressure, cholesterol, exercise, and diet goals."
        )
    else:  # High
        base.append(
            "It may be important to seek professional medical advice to review "
            "your risk factors and next steps."
        )
    return base


def generate_explanation(
    inputs: Dict[str, Any],
    risk_score: float,
    risk_level: str,
    top_features: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """Return a natural-language explanation of the prediction using Gemini.

    Falls back to a simple template if the API key is not configured or if
    the Gemini call fails.
    """
    if not _get_client():
        return _basic_explanation_fallback(risk_level, risk_score)

    prompt = f"""
    You are helping to explain the output of a heart-disease risk prediction
    tool. The tool is not a diagnostic system and must not give medical
    advice. Explain the result in clear, simple language (about 3–5
    sentences). Avoid technical machine-learning terms.

    IMPORTANT GUIDELINES:
    - Do NOT claim to diagnose or treat any condition.
    - Emphasize that the result is only an estimate based on limited inputs.
    - Encourage the person to consult a qualified healthcare professional.
    - Use neutral, supportive language.

    Language: {language}

    Inputs (key=value): {inputs}
    Risk score (0–1): {risk_score:.3f}
    Risk level: {risk_level}
    Top contributing features (name, shap_value): {top_features}
    """.strip()

    try:
        model = genai.GenerativeModel(_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or "".join(
            [p.text for p in getattr(resp, "candidates", []) if hasattr(p, "text")]
        )
        text = text.strip() if text else ""
        if not text:
            return _basic_explanation_fallback(risk_level, risk_score)
        return text
    except Exception:
        return _basic_explanation_fallback(risk_level, risk_score)


def generate_lifestyle_suggestions(
    inputs: Dict[str, Any],
    risk_score: float,
    risk_level: str,
    top_features: List[Dict[str, Any]],
    language: str = "en",
) -> List[str]:
    """Return a list of high-level lifestyle suggestions using Gemini.

    Suggestions must be framed as general educational information, not
    medical advice. Falls back to a small rule-based template if Gemini is
    unavailable.
    """
    if not _get_client():
        return _basic_lifestyle_fallback(risk_level)

    prompt = f"""
    You are assisting with a heart-health education tool. Based on the
    following estimated risk and risk factors, provide 3–5 short,
    high-level lifestyle suggestions that a person could discuss with a
    healthcare professional. These should be generic, non-personalized
    tips about heart-healthy habits.

    STRICT RULES:
    - Do NOT give any diagnosis.
    - Do NOT mention specific medications or treatment plans.
    - Do NOT sound certain about the person's actual health.
    - Emphasize that the suggestions are general and should be discussed
      with a doctor.
    - Keep each suggestion to 1–2 sentences.

    Language: {language}

    Inputs (key=value): {inputs}
    Risk score (0–1): {risk_score:.3f}
    Risk level: {risk_level}
    Top contributing features (name, shap_value): {top_features}
    """.strip()

    try:
        model = genai.GenerativeModel(_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        lines = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
        cleaned = [ln for ln in lines if ln]
        return cleaned or _basic_lifestyle_fallback(risk_level)
    except Exception:
        return _basic_lifestyle_fallback(risk_level)


def generate_followup_plan(
    inputs: Dict[str, Any],
    risk_score: float,
    risk_level: str,
    top_features: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """Return a short, non-medical follow-up plan to discuss with a doctor.

    This is intentionally NOT a prescription: it focuses on questions to ask,
    topics to review, and generic next steps to consider with a clinician.
    Falls back to a simple template if Gemini is unavailable.
    """
    if not _get_client():
        return (
            "Consider scheduling an appointment with a qualified healthcare "
            "professional to review your blood pressure, cholesterol, "
            "lifestyle, and any symptoms. Bring this assessment result and "
            "ask what additional tests or monitoring they recommend."
        )

    prompt = f"""
    You are assisting with a heart-health education tool. Based on the
    estimated risk and contributing factors, write a brief follow-up plan
    that a person could discuss with a qualified healthcare professional.

    The plan should:
    - Be written in {language}.
    - NOT be a medical prescription or treatment plan.
    - Suggest questions to ask a doctor, topics to review, or possible
      next steps (e.g., ask about further tests, monitoring, or lifestyle
      changes).
    - Clearly encourage the reader to consult a doctor for any decisions.
    - Be 4–7 short bullet points.

    STRICT RULES:
    - Do NOT name specific drugs or dosages.
    - Do NOT claim to diagnose or cure any disease.
    - Do NOT instruct the user to start or stop medications.

    Inputs (key=value): {inputs}
    Risk score (0–1): {risk_score:.3f}
    Risk level: {risk_level}
    Top contributing features (name, shap_value): {top_features}
    """.strip()

    try:
        model = genai.GenerativeModel(_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        text = text.strip()
        if not text:
            return (
                "Consider discussing this assessment with a doctor, asking "
                "about further evaluation, lifestyle options, and how often "
                "your heart health should be monitored."
            )
        return text
    except Exception:
        return (
            "Consider scheduling an appointment with a qualified healthcare "
            "professional to review your results, ask about further tests, "
            "and discuss safe lifestyle options." 
        )


def generate_prescription_summary(
    inputs: Dict[str, Any],
    risk_score: float,
    risk_level: str,
    top_features: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """Return a structured, prescription-style summary for doctor review.

    This is *not* a real prescription: it does not include any medication
    names, drug classes, or dosages. It outlines clinical focus areas,
    lifestyle themes, follow-up evaluations, and safety reminders that a
    doctor could review with the patient.
    """
    if not _get_client():
        return (
            "This summary is intended to help structure a discussion with "
            "a qualified healthcare professional. It does not recommend "
            "specific medications or doses. Please consult your doctor for "
            "any treatment decisions."
        )

    # Format inputs as key=value lines to keep the prompt general.
    kv_pairs = ", ".join(f"{k}={v}" for k, v in inputs.items())

    prompt = f"""
    You are helping with a heart-health education tool. Based on the
    following patient data and estimated risk, write a brief, structured
    summary that a DOCTOR could use as a starting point for their own
    prescription and management plan.

    SAFETY RULES (MUST FOLLOW):
    - Do NOT name any medications or drug classes.
    - Do NOT suggest any dosages, frequencies, or treatment durations.
    - Do NOT instruct the patient to start, stop, or change medication.
    - Do NOT claim to diagnose, cure, or prevent disease.
    - Keep all content as general topics or areas for the doctor to
      consider and discuss.

    Patient inputs (key=value): {kv_pairs}
    Estimated heart disease risk (0–1): {risk_score:.3f}
    Risk level: {risk_level}
    Top contributing model features: {top_features}

    Write the summary in {language} with the following sections:

    1) Clinical focus areas for the doctor
       - Bulleted list of risk factors or symptoms that may merit review.

    2) Lifestyle focus areas
       - General lifestyle domains (e.g., activity, diet patterns, smoking)
         to discuss, without personalized instructions.

    3) Possible follow-up evaluations
       - General examples of tests or referrals a doctor might consider
         (e.g., further cardiac evaluation), without ordering them.

    4) Safety reminder for the patient
       - Short paragraph reinforcing that only their own doctor can
         prescribe medication, decide on doses, or choose treatment.
    """.strip()

    try:
        model = genai.GenerativeModel(_MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        text = text.strip()
        if not text:
            return (
                "Discuss these results with a qualified doctor, who can "
                "review risk factors, advise on lifestyle, and decide if "
                "any tests or treatments are appropriate for you."
            )
        return text
    except Exception:
        return (
            "This high-level summary is intended for discussion with a "
            "healthcare professional and does not include medication names "
            "or dosages. Please consult your doctor for individual advice."
        )
