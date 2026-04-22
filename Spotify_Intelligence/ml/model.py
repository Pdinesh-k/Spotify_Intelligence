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


def _generate_synthetic_data(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n_engaged = n // 2
    n_churned = n - n_engaged

    # Engaged listeners: healthy signal profile
    engaged = np.column_stack([
        rng.normal(-0.04, 0.05, n_engaged).clip(-0.30, 0.08),   # skip_rate_trend (low/stable)
        rng.normal(0.5, 1.2, n_engaged),                         # session_freq_delta (growing)
        rng.normal(0.78, 0.10, n_engaged).clip(0.45, 1.0),      # listen_depth (high)
        rng.normal(-0.08, 0.12, n_engaged).clip(-0.40, 0.20),   # genre_entropy_drop (stable)
        rng.normal(1.2, 1.0, n_engaged).clip(0.0, 5.0),         # time_of_day_shift (small)
        rng.normal(0.8, 1.0, n_engaged).clip(0.0, 7.0),         # days_new_artist (low)
        rng.normal(0.25, 0.10, n_engaged).clip(0.0, 0.60),      # repeat_play_ratio (moderate)
    ])

    # Churning listeners: disengagement signal profile
    churned = np.column_stack([
        rng.normal(0.22, 0.10, n_churned).clip(0.05, 0.55),     # skip_rate_trend (rising)
        rng.normal(-2.2, 1.2, n_churned),                        # session_freq_delta (falling)
        rng.normal(0.33, 0.10, n_churned).clip(0.10, 0.55),     # listen_depth (shallow)
        rng.normal(0.35, 0.15, n_churned).clip(0.0, 0.80),      # genre_entropy_drop (collapsing)
        rng.normal(4.5, 2.0, n_churned).clip(0.0, 11.0),        # time_of_day_shift (large)
        rng.normal(5.8, 1.0, n_churned).clip(0.0, 7.0),         # days_new_artist (high)
        rng.normal(0.68, 0.10, n_churned).clip(0.35, 1.0),      # repeat_play_ratio (high)
    ])

    X = np.vstack([engaged, churned])
    y = np.array([0] * n_engaged + [1] * n_churned)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


_XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
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

    # Calibrated model uses cv=3 on the full dataset so it has enough data per fold
    calibrated = CalibratedClassifierCV(XGBClassifier(**_XGB_PARAMS), cv=3, method="isotonic")
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
