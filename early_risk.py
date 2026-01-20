# early_risk.py
import numpy as np


def _safe_get(row, col, default=None):
    """Safely read a column value from a pandas row/dict-like."""
    try:
        val = row.get(col, default)
        if val is None:
            return default
        return val
    except Exception:
        return default


def compute_early_risk(row):
    """
    Rule-based early attrition risk scoring.

    Input:
      row: pandas Series (one employee row)

    Output:
      (score_0_100, bucket, explanation_string)
    """

    score = 0
    reasons = []

    # ---- Drivers ----
    years_at_company = _safe_get(row, "YearsAtCompany", None)
    job_level = _safe_get(row, "JobLevel", None)
    training_times = _safe_get(row, "TrainingTimesLastYear", None)

    overtime = _safe_get(row, "OverTime", None)               # Yes/No or 1/0
    years_in_role = _safe_get(row, "YearsInCurrentRole", None)

    # -------------------------------
    # 1) YearsAtCompany (high weight)
    # -------------------------------
    if years_at_company is not None:
        try:
            y = float(years_at_company)
            if y <= 1:
                score += 35
                reasons.append("Very new employee (YearsAtCompany ≤ 1)")
            elif y <= 3:
                score += 25
                reasons.append("Early tenure (YearsAtCompany 2–3)")
            elif y <= 5:
                score += 10
                reasons.append("Moderate tenure (YearsAtCompany 4–5)")
            else:
                score += 0
        except Exception:
            pass

    # -------------------------------
    # 2) JobLevel (low job level => more risk)
    # -------------------------------
    if job_level is not None:
        try:
            jl = float(job_level)
            if jl <= 1:
                score += 20
                reasons.append("Low job level (JobLevel ≤ 1)")
            elif jl <= 2:
                score += 10
                reasons.append("Mid-low job level (JobLevel = 2)")
            else:
                score += 0
        except Exception:
            pass

    # -------------------------------
    # 3) TrainingTimesLastYear (low training => risk)
    # -------------------------------
    if training_times is not None:
        try:
            t = float(training_times)
            if t == 0:
                score += 20
                reasons.append("No training last year (TrainingTimesLastYear = 0)")
            elif t <= 1:
                score += 10
                reasons.append("Very low training (TrainingTimesLastYear ≤ 1)")
            else:
                score += 0
        except Exception:
            pass

    # -------------------------------
    # Optional add-ons (if present)
    # -------------------------------
    # OverTime -> risk up
    if overtime is not None:
        try:
            ov = str(overtime).strip().lower()
            if ov in ["yes", "1", "true"]:
                score += 10
                reasons.append("OverTime = Yes")
        except Exception:
            pass

    # YearsInCurrentRole: if stuck long time -> mild risk
    if years_in_role is not None:
        try:
            yr = float(years_in_role)
            if yr >= 5:
                score += 5
                reasons.append("Long time in same role (YearsInCurrentRole ≥ 5)")
        except Exception:
            pass

    # Clamp 0..100
    score = int(np.clip(score, 0, 100))

    # Bucket
    if score < 35:
        bucket = "Low"
    elif score < 70:
        bucket = "Medium"
    else:
        bucket = "High"

    explanation = "; ".join(reasons) if reasons else "No major early-risk signals triggered"

    return score, bucket, explanation
