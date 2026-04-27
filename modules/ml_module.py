"""
InsOps · Machine Learning Module
Auto train/test split, model selection, evaluation, feature importance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")


def run_ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    model_type: str,
    task_type: str,
) -> dict:
    """
    Returns a result dict with score, predictions, feature importances.
    """
    try:
        # Drop non-numeric cols that remain (high-card strings weren't encoded)
        df = df.select_dtypes(include=np.number).copy()

        if target_col not in df.columns:
            return {"error": f"Target column `{target_col}` not found after encoding."}

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if X.shape[1] == 0:
            return {"error": "No feature columns available after removing target."}

        # Drop rows where target is NaN
        mask = y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 20:
            return {"error": f"Not enough rows ({len(X)}) to train a model. Need ≥ 20."}

        # For classification, ensure discrete labels
        if task_type == "Classification":
            y = y.round().astype(int)
            n_classes = y.nunique()
            if n_classes > 30:
                return {"error": f"Too many classes ({n_classes}) for classification. Switch to Regression."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Select model
        if task_type == "Classification":
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = LogisticRegression(max_iter=500, random_state=42)
        else:
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = Ridge(alpha=1.0)

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        # Metrics
        if task_type == "Classification":
            avg = "binary" if y.nunique() == 2 else "weighted"
            score = accuracy_score(y_test, preds)
            score_label = "Accuracy"
        else:
            score = r2_score(y_test, preds)
            score_label = "R² Score"

        # Feature importance
        fi_df = None
        try:
            if hasattr(model, "feature_importances_"):
                fi_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
            elif hasattr(model, "coef_"):
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                fi_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": np.abs(coef),
                }).sort_values("importance", ascending=False)
        except Exception:
            pass

        # Predictions df
        pred_df = pd.DataFrame({
            "actual":    y_test.values[:50],
            "predicted": preds[:50],
        })
        if task_type == "Regression":
            pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]

        return {
            "score":             round(score, 4),
            "score_label":       score_label,
            "train_size":        len(X_train),
            "test_size":         len(X_test),
            "predictions_df":    pred_df,
            "feature_importance": fi_df,
            "model_name":        type(model).__name__,
        }

    except Exception as e:
        return {"error": str(e)}
