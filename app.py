# app_fixed.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import traceback
import json
from typing import Dict, Any

app = FastAPI()

# Serve index.html if present
@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<html><body><h3>FastAPI ML service</h3></body></html>"

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Utility to encode charts to base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Categorical encoding
def encode_categorical_train(df: pd.DataFrame):
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    # Only object / categorical columns (exclude numeric-like)
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        # Fill NaN temporarily with a string token so fit_transform doesn't error
        df[col] = df[col].fillna("__MISSING__")
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def encode_categorical_future(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            # Fill missing with the same token used in training if possible
            df[col] = df[col].fillna("__MISSING__").astype(str)
            # If unseen label -> LabelEncoder will error; map unseen to a special value if possible
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # handle unseen categories by adding them to the encoder classes
                existing = list(le.classes_)
                new_vals = sorted(set(df[col].unique()) - set(existing))
                if new_vals:
                    le.classes_ = np.array(existing + new_vals, dtype=object)
                df[col] = le.transform(df[col])
    return df

# Save model metadata (feature order, encoders)
def save_metadata(metadata: Dict[str, Any], path="models/metadata.pkl"):
    joblib.dump(metadata, path)

def load_metadata(path="models/metadata.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# ---------------- TRAIN API ----------------
@app.post("/train")
async def train(
    india_present_csv: UploadFile = File(...),
    global_present_csv: UploadFile = File(...)
   
):
    try:
        # Read CSVs
        india_present = pd.read_csv(india_present_csv.file)
        india_present, india_encoders = encode_categorical_train(india_present)

        

        global_present = pd.read_csv(global_present_csv.file)
        global_present, global_encoders = encode_categorical_train(global_present)

       

        # Prepare features and target (training on india_present as before)
        X = india_present.drop("diabetes", axis=1, errors="ignore")
        if "diabetes" in india_present.columns:
            y = india_present["diabetes"].copy()
            if y.dtype == "object":
                y = y.map({"Yes": 1, "No": 0}).astype(float)
            mask = y.notna()
            y = y.loc[mask].astype(int)
            X = X.loc[mask]
        else:
            y = np.random.randint(0, 2, len(india_present))

        # Handle NaNs in features
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "__MISSING__")

        # Keep feature order for prediction later
        feature_names = X.columns.tolist()

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train multiple models
        models = {
            "Random_Forest": RandomForestClassifier(random_state=100),
            "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=100),
            "Decision_Tree": DecisionTreeClassifier(random_state=100),
            "SVM": SVC(probability=True, random_state=100)
        }

        results = {}
        accuracy = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            # Some models might not have predict_proba (we set probability=True for SVC above),
            # but guard anyway:
            if hasattr(model, "predict_proba"):
                preds_proba = model.predict_proba(X_test)[:, 1]
            else:
                # fallback: use decision_function -> map to probs with logistic-like transform
                df_scores = model.decision_function(X_test)
                # simple sigmoid
                preds_proba = 1.0 / (1.0 + np.exp(-df_scores))

            # compute metrics safely (handle cases where y_test has only one class)
            try:
                results[name] = {
                    "accuracy": round(accuracy_score(y_test, preds), 3),
                    "precision": round(precision_score(y_test, preds, zero_division=0), 3),
                    "recall": round(recall_score(y_test, preds, zero_division=0), 3),
                    "f1_score": round(f1_score(y_test, preds, zero_division=0), 3),
                    "roc_auc": round(roc_auc_score(y_test, preds_proba), 3) if len(np.unique(y_test)) > 1 else None,
                }
            except Exception:
                results[name] = {"error": "Could not compute metrics (maybe single-class in y_test)"}

            accuracy[name] = results[name].get("accuracy")

            # Save model file
            model_path = f"models/{name}.pkl"
            joblib.dump(model, model_path)

        # Save metadata for predict: which features, and india_encoders
        metadata = {
            "feature_names": feature_names,
            "encoders": india_encoders  # these are LabelEncoder objects
        }
        save_metadata(metadata, path="models/metadata.pkl")

        # Generate pie chart (only from numeric series of the target distribution if exists)
        fig, ax = plt.subplots()
        if "diabetes" in india_present.columns:
            india_present["diabetes"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        else:
            # fallback: random pie
            pd.Series([0.5, 0.5], index=["No data", "No data"]).plot.pie(autopct="%1.1f%%", ax=ax)
        pie_chart_base64 = fig_to_base64(fig)
        plt.close(fig)

        # Bar chart: only numeric columns mean
        fig, ax = plt.subplots()
        numeric_means = X_train.select_dtypes(include=[np.number]).mean()
        if numeric_means.empty:
            numeric_means = pd.Series([0])
        numeric_means.plot.bar(ax=ax)
        bar_chart_base64 = fig_to_base64(fig)
        plt.close(fig)

        charts = {
            "india": {
                "present": {"pie_chart": pie_chart_base64}
                
            },
            "global": {
                "present": {"pie_chart": pie_chart_base64}
                
            }
        }

        return {
            "message": "Models trained successfully",
            "accuracy": accuracy,
            "results": results,
            "charts": charts,
            "feature_names": feature_names  # return so client can prepare correct order
        }

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})

# ---------------- PREDICT API ----------------
@app.post("/predict")
async def predict(
    region: str = Form(...),
    data_type: str = Form(...),
    algorithm: str = Form(...),
    features: str = Form(...)
):
    """
    Predict endpoint accepts:
    - features: either a comma-separated string of feature values in the same order as training feature_names,
                OR a JSON string representing a dict { "feature_name": value, ... }.
    - algorithm: name used during training (e.g. Random_Forest, Logistic_Regression, Decision_Tree, SVM)
    """
    try:
        # Load model
        model_file = f"models/{algorithm}.pkl"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model '{algorithm}' not found. Train first (names are: Random_Forest, Logistic_Regression, Decision_Tree, SVM).")

        model = joblib.load(model_file)

        # Load metadata
        metadata = load_metadata("models/metadata.pkl")
        if metadata is None:
            raise FileNotFoundError("Metadata not found (models/metadata.pkl). Retrain to generate metadata.")

        feature_names = metadata.get("feature_names", [])
        encoders = metadata.get("encoders", {})

        # Parse features input: either JSON dict or comma-separated
        parsed_input = None
        features = features.strip()
        if (features.startswith("{") and features.endswith("}")) or (features.startswith("[") and features.endswith("]")):
            # try JSON parse
            parsed = json.loads(features)
            if isinstance(parsed, dict):
                # ensure all feature_names exist
                missing = [f for f in feature_names if f not in parsed]
                if missing:
                    raise ValueError(f"Missing features in input JSON: {missing}")
                # Build list in order
                parsed_input = [parsed[f] for f in feature_names]
            elif isinstance(parsed, list):
                parsed_input = parsed
            else:
                raise ValueError("JSON features must be list or dict.")
        else:
            # comma-separated
            parsed_input = [x.strip() for x in features.split(",") if x.strip() != ""]

        # If parsed_input is dict keyed, above handled; now ensure length matches or try to cast to numeric/dtype handling
        if isinstance(parsed_input, list) and len(parsed_input) != len(feature_names):
            raise ValueError(f"Number of feature values ({len(parsed_input)}) does not match expected ({len(feature_names)}). Feature order: {feature_names}")

        # Convert to DataFrame row so we can apply encoders easily
        row = pd.DataFrame([parsed_input], columns=feature_names)

        # Convert numeric-like values to numeric where appropriate
        for col in row.columns:
            # if the training data had numeric dtype for this column, try to cast to numeric
            try:
                row[col] = pd.to_numeric(row[col])
            except Exception:
                # keep as string for categorical columns
                row[col] = row[col].astype(str)

        # Apply encoders for categorical columns
        row = encode_categorical_future(row, encoders)

        # Ensure ordering and numeric types
        X_row = row[feature_names].astype(float).values

        # Predict probability
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(X_row)[0]
        else:
            # fallback: decision_function -> softmax-ish
            scores = model.decision_function(X_row)
            # if binary, scores is scalar; convert
            if np.ndim(scores) == 0:
                scores = np.array([scores, -scores])
            # softmax
            exp = np.exp(scores - np.max(scores))
            pred_prob = (exp / exp.sum()).astype(float)

        pred_class = int(model.predict(X_row)[0])
        response = {
            "prediction": pred_class,
            "probability": {
                "No Diabetes": float(pred_prob[0]),
                "Diabetes": float(pred_prob[1]) if len(pred_prob) > 1 else float(1.0 - pred_prob[0])
            },
            "confidence": float(max(pred_prob)),
            "feature_order": feature_names
        }
        return response

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})