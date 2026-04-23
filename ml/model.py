import os
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import MODEL_PATH
from ml.features import FEATURE_NAMES

RANDOM_STATE = 42


def _generate_synthetic_data(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n_engaged = n // 2
    n_churned = n - n_engaged

    # Feature ranges calibrated to real Spotify API output ranges:
    #   skip_rate_trend:    0.00 – 0.25   (popularity proxy)
    #   session_freq_delta: -3.0 – 3.0    (sessions/day delta from timestamps)
    #   listen_depth:       0.45 – 0.80   (avg_energy from audio features)
    #   genre_entropy_drop: 0.00 – 1.00   (genre concentration)
    #   time_of_day_shift:  0.0  – 10.0   (hour drift from timestamps)
    #   days_new_artist:    0.0  – 7.0    (days since new artist / top-artist fallback)
    #   repeat_play_ratio:  0.0  – 0.80   (recent vs all-time overlap)
    #
    # Wide standard deviations = lots of class overlap = realistic intermediate probabilities

    engaged = np.column_stack([
        rng.normal(0.01, 0.07, n_engaged).clip(-0.10, 0.15),    # skip_rate_trend low
        rng.normal(0.8, 2.0, n_engaged),                          # session_freq_delta positive
        rng.normal(0.68, 0.14, n_engaged).clip(0.40, 1.0),       # listen_depth ~0.68
        rng.normal(0.18, 0.18, n_engaged).clip(0.0, 0.65),       # genre_entropy_drop low
        rng.normal(2.0, 2.2, n_engaged).clip(0.0, 9.0),          # time_of_day_shift small
        rng.normal(2.0, 2.0, n_engaged).clip(0.0, 7.0),          # days_new_artist low
        rng.normal(0.25, 0.18, n_engaged).clip(0.0, 0.75),       # repeat_play_ratio moderate
    ])

    churned = np.column_stack([
        rng.normal(0.12, 0.08, n_churned).clip(0.0, 0.35),       # skip_rate_trend rising
        rng.normal(-1.2, 2.0, n_churned),                         # session_freq_delta negative
        rng.normal(0.56, 0.14, n_churned).clip(0.25, 0.85),      # listen_depth ~0.56 (overlaps!)
        rng.normal(0.55, 0.22, n_churned).clip(0.10, 1.0),       # genre_entropy_drop high
        rng.normal(5.5, 2.5, n_churned).clip(0.0, 12.0),         # time_of_day_shift large
        rng.normal(5.0, 1.8, n_churned).clip(0.0, 7.0),          # days_new_artist high
        rng.normal(0.55, 0.20, n_churned).clip(0.0, 1.0),        # repeat_play_ratio high
    ])

    X = np.vstack([engaged, churned])
    y = np.array([0] * n_engaged + [1] * n_churned)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


_XGB_PARAMS = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=0.5,
    reg_alpha=1.0,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)


def train_and_save() -> tuple[XGBClassifier, CalibratedClassifierCV]:
    X, y = _generate_synthetic_data()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Base model fitted on training split — used for SHAP (needs a fitted XGB)
    base = XGBClassifier(**_XGB_PARAMS)
    base.fit(X_train, y_train)

    # Sigmoid calibration (Platt scaling) gives smooth continuous probabilities
    # unlike isotonic which produces a step function and snaps to 0/100
    calibrated = CalibratedClassifierCV(XGBClassifier(**_XGB_PARAMS), cv=5, method="sigmoid")
    calibrated.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"base": base, "calibrated": calibrated}, f)

    return base, calibrated


def load_models() -> tuple[XGBClassifier, CalibratedClassifierCV]:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle["base"], bundle["calibrated"]
    return train_and_save()


def predict(features: dict) -> dict:
    """
    Run inference + SHAP on a feature dict.

    Returns:
      churn_probability  – calibrated float in [0, 1]
      risk_level         – 'Low' | 'Medium' | 'High'
      shap_values        – dict mapping feature name → SHAP value
      top_drivers        – top 3 (feature, shap_value) tuples by |shap|
      feature_values     – raw feature values used for inference
    """
    base, calibrated = load_models()

    X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    prob = float(calibrated.predict_proba(X)[0][1])

    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(df)
    # XGBClassifier returns a single 2D array (not list) in recent versions
    sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

    shap_dict = {f: float(v) for f, v in zip(FEATURE_NAMES, sv)}
    top_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        "churn_probability": prob,
        "risk_level": "High" if prob > 0.65 else "Medium" if prob > 0.38 else "Low",
        "shap_values": shap_dict,
        "top_drivers": top_drivers,
        "feature_values": {f: features.get(f, 0.0) for f in FEATURE_NAMES},
        "base_value": float(explainer.expected_value),
    }
