# early_risk.py
import numpy as np


def _safe_get(row, col, default=None):
    try:
        val = row.get(col, default)
        return default if val is None else val
    except Exception:
        return default


def compute_early_risk(row):
    score = 0
    reasons = []

    years_at_company = _safe_get(row, "YearsAtCompany", None)
    job_level = _safe_get(row, "JobLevel", None)
    training_times = _safe_get(row, "TrainingTimesLastYear", None)
    overtime = _safe_get(row, "OverTime", None)
    years_in_role = _safe_get(row, "YearsInCurrentRole", None)

    # 1) YearsAtCompany (high weight)
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
    except Exception:
        pass

    # 2) JobLevel (low level => more risk)
    try:
        jl = float(job_level)
        if jl <= 1:
            score += 20
            reasons.append("Low job level (JobLevel ≤ 1)")
        elif jl <= 2:
            score += 10
            reasons.append("Mid-low job level (JobLevel = 2)")
    except Exception:
        pass

    # 3) TrainingTimesLastYear (low training => risk)
    try:
        t = float(training_times)
        if t == 0:
            score += 20
            reasons.append("No training last year (TrainingTimesLastYear = 0)")
        elif t <= 1:
            score += 10
            reasons.append("Very low training (TrainingTimesLastYear ≤ 1)")
    except Exception:
        pass

    # OverTime -> risk up
    try:
        ov = str(overtime).strip().lower()
        if ov in ["yes", "1", "true"]:
            score += 10
            reasons.append("OverTime = Yes")
    except Exception:
        pass

    # YearsInCurrentRole -> mild risk if too long
    try:
        yr = float(years_in_role)
        if yr >= 5:
            score += 5
            reasons.append("Long time in same role (YearsInCurrentRole ≥ 5)")
    except Exception:
        pass

    score = int(np.clip(score, 0, 100))

    if score < 35:
        bucket = "Low"
    elif score < 70:
        bucket = "Medium"
    else:
        bucket = "High"

    explanation = "; ".join(reasons) if reasons else "No major early-risk signals triggered"
    return score, bucket, explanation
