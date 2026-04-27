"""
InsOps · Advanced Machine Learning Module
- XGBoost, LightGBM, Random Forest, Logistic/Ridge, Decision Tree, KNN, SVM
- Auto model leaderboard
- 5-fold cross-validation
- SHAP explanations
- Live prediction input
- Model export (.pkl)
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_auc_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def get_available_models(task_type: str) -> dict:
    models = {}
    if task_type == "Classification":
        models["Random Forest"]        = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        models["Decision Tree"]        = DecisionTreeClassifier(max_depth=6, random_state=42)
        models["Logistic Regression"]  = LogisticRegression(max_iter=500, random_state=42)
        models["KNN"]                  = KNeighborsClassifier(n_neighbors=5)
        models["Gradient Boosting"]    = GradientBoostingClassifier(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models["XGBoost"]          = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
        if LGBM_AVAILABLE:
            models["LightGBM"]         = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models["Random Forest"]        = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        models["Decision Tree"]        = DecisionTreeRegressor(max_depth=6, random_state=42)
        models["Ridge Regression"]     = Ridge(alpha=1.0)
        models["KNN"]                  = KNeighborsRegressor(n_neighbors=5)
        models["Gradient Boosting"]    = GradientBoostingRegressor(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models["XGBoost"]          = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if LGBM_AVAILABLE:
            models["LightGBM"]         = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    return models


def _prep_data(df, target_col, test_size, task_type):
    df = df.select_dtypes(include=np.number).copy()
    if target_col not in df.columns:
        return None, f"Target `{target_col}` not found after encoding."

    X = df.drop(columns=[target_col])
    y = df[target_col]
    mask = y.notna()
    X, y = X[mask], y[mask]

    if len(X) < 30:
        return None, f"Not enough rows ({len(X)}). Need ≥ 30."

    if task_type == "Classification":
        y = y.round().astype(int)
        if y.nunique() > 30:
            return None, f"Too many classes ({y.nunique()}) for classification."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task_type == "Classification" else None
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "scaler": scaler, "feature_names": X.columns.tolist()
    }, None


def _score_model(model, X_train_s, X_test_s, y_train, y_test, task_type, cv_folds, X, y):
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    if task_type == "Classification":
        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        metrics = {
            "Accuracy":  round(accuracy_score(y_test, preds), 4),
            "F1 Score":  round(f1_score(y_test, preds, average=avg, zero_division=0), 4),
            "Precision": round(precision_score(y_test, preds, average=avg, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, preds, average=avg, zero_division=0), 4),
        }
        try:
            if len(np.unique(y)) == 2:
                proba = model.predict_proba(X_test_s)[:, 1]
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, proba), 4)
        except Exception:
            pass
        primary_score = metrics["Accuracy"]
        cv_scoring    = "accuracy"
        cm = confusion_matrix(y_test, preds)
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics = {
            "R² Score": round(r2_score(y_test, preds), 4),
            "RMSE":     round(rmse, 4),
            "MAE":      round(mean_absolute_error(y_test, preds), 4),
        }
        primary_score = metrics["R² Score"]
        cv_scoring    = "r2"
        cm = None

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) \
         if task_type == "Classification" else \
         KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scaler_cv = StandardScaler()
    X_scaled  = scaler_cv.fit_transform(X)
    cv_scores = cross_val_score(model.__class__(**model.get_params()),
                                X_scaled, y, cv=cv, scoring=cv_scoring, n_jobs=-1)
    metrics[f"CV Mean ({cv_folds}-fold)"]  = round(cv_scores.mean(), 4)
    metrics[f"CV Std ({cv_folds}-fold)"]   = round(cv_scores.std(), 4)

    return metrics, primary_score, preds, cm


def run_single_model(df, target_col, model_name, task_type, test_size=0.2, cv_folds=5,
                     hyperparams=None):
    data, err = _prep_data(df, target_col, test_size, task_type)
    if err:
        return {"error": err}

    models = get_available_models(task_type)
    if model_name not in models:
        return {"error": f"Model '{model_name}' not available."}

    model = models[model_name]
    if hyperparams:
        try:
            model.set_params(**hyperparams)
        except Exception:
            pass

    metrics, primary_score, preds, cm = _score_model(
        model,
        data["X_train_s"], data["X_test_s"],
        data["y_train"],   data["y_test"],
        task_type, cv_folds, data["X"], data["y"]
    )

    # Feature importance
    fi_df = None
    try:
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature":    data["feature_names"],
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
        elif hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            fi_df = pd.DataFrame({
                "feature":    data["feature_names"],
                "importance": np.abs(coef),
            }).sort_values("importance", ascending=False)
    except Exception:
        pass

    # SHAP values
    shap_vals = None
    try:
        import shap
        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                    else shap.LinearExplainer(model, data["X_train_s"])
        shap_vals = explainer.shap_values(data["X_test_s"][:100])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception:
        pass

    # Predictions df
    pred_df = pd.DataFrame({
        "actual":    data["y_test"].values[:50],
        "predicted": preds[:50],
    })
    if task_type == "Regression":
        pred_df["residual"] = (pred_df["actual"] - pred_df["predicted"]).round(2)

    # Pickle model
    model_bytes = pickle.dumps({
        "model":   model,
        "scaler":  data["scaler"],
        "features": data["feature_names"],
        "task":    task_type,
        "target":  target_col,
    })

    return {
        "model_name":        model_name,
        "metrics":           metrics,
        "primary_score":     primary_score,
        "predictions_df":    pred_df,
        "feature_importance": fi_df,
        "shap_values":       shap_vals,
        "shap_features":     data["feature_names"],
        "X_test_sample":     data["X_test"][:100],
        "confusion_matrix":  cm,
        "train_size":        len(data["X_train"]),
        "test_size":         len(data["X_test"]),
        "task_type":         task_type,
        "model_bytes":       model_bytes,
    }


def run_leaderboard(df, target_col, task_type, test_size=0.2, cv_folds=5):
    """Train ALL available models and return a ranked leaderboard."""
    data, err = _prep_data(df, target_col, test_size, task_type)
    if err:
        return None, err

    models  = get_available_models(task_type)
    results = []

    for name, model in models.items():
        try:
            metrics, primary_score, preds, cm = _score_model(
                model,
                data["X_train_s"], data["X_test_s"],
                data["y_train"],   data["y_test"],
                task_type, cv_folds, data["X"], data["y"]
            )
            row = {"Model": name, "Primary Score": primary_score}
            row.update(metrics)
            results.append(row)
        except Exception as e:
            results.append({"Model": name, "Primary Score": -999, "Error": str(e)})

    lb_df = pd.DataFrame(results).sort_values("Primary Score", ascending=False).reset_index(drop=True)
    lb_df.index += 1  # rank from 1
    return lb_df, None


def predict_single_row(model_bytes: bytes, input_dict: dict) -> str:
    """Run inference on a single row from user input."""
    obj    = pickle.loads(model_bytes)
    model  = obj["model"]
    scaler = obj["scaler"]
    feats  = obj["features"]

    row = pd.DataFrame([{f: input_dict.get(f, 0) for f in feats}])
    row_s = scaler.transform(row)
    pred  = model.predict(row_s)[0]
    return pred
