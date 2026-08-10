from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
import torch
import joblib
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
from architectures import RNNModel, LSTMModel
from data_utils import load_and_clean_data, add_features
from paths import DATA_PATH, MODELS_DIR

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS, DF_FULL, TARGET_SCALER, FEAT_SCALER = {}, None, None, None

class SpecificPredictionInput(BaseModel):
    data: list
    model_name: str

@app.on_event("startup")
def startup_event():
    global DF_FULL, TARGET_SCALER, FEAT_SCALER, MODELS
    try:
        TARGET_SCALER = joblib.load(MODELS_DIR / "target_scaler.pkl")
        FEAT_SCALER = joblib.load(MODELS_DIR / "feat_scaler.pkl")
        for m_type in ["LSTM", "RNN"]:
            m_class = LSTMModel if m_type == "LSTM" else RNNModel
            m = m_class(input_size=19, hidden_size=64, num_layers=2)
            m.load_state_dict(torch.load(MODELS_DIR / f"{m_type.lower()}_model.pth", map_location="cpu"))
            m.eval()
            MODELS[m_type] = m
        raw_df = load_and_clean_data(DATA_PATH)
        ts = raw_df["datetime"].dt.strftime("%b %d, %Y - %H:%M").tolist()
        eng_df = add_features(raw_df)
        eng_df["datetime_str"] = ts
        DF_FULL = eng_df
        print("Backend Ready for Comparison Mode.")
    except Exception as e:
        print(f" Error: {e}")

# Helper function to run the math
def run_model_math(model_name, window_df):
    target_df = window_df[["nat_demand"]]
    scaled_target = TARGET_SCALER.transform(target_df)
    feat_df = window_df.drop(columns=["nat_demand", "datetime_str"])
    scaled_feats = FEAT_SCALER.transform(feat_df)
    combined = np.hstack([scaled_target, scaled_feats])
    input_tensor = torch.tensor(combined).float().unsqueeze(0)
    model = MODELS.get(model_name.upper(), MODELS["LSTM"])
    with torch.no_grad():
        pred_scaled = model(input_tensor)
    res_mw = TARGET_SCALER.inverse_transform(pred_scaled.numpy())
    return float(res_mw[0][0])

@app.get("/predict_sample/{model_name}")
async def predict_sample(model_name: str):
    try:
        idx = np.random.randint(int(len(DF_FULL)*0.8), len(DF_FULL)-25)
        window_df = DF_FULL.iloc[idx : idx + 24].copy()
        prediction = run_model_math(model_name, window_df)
        
        return {
            "history": window_df["nat_demand"].tolist(),
            "prediction": prediction,
            "timestamp": window_df.iloc[-1]["datetime_str"],
            "raw_features": window_df.to_json() # Send the window back so Flutter can store it
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import io

@app.post("/predict_specific")
async def predict_specific(payload: dict):
    try:
        # Reconstruct DF from the JSON string sent by the dashboard.
        # pd.read_json() treats a bare string as a file PATH, not JSON
        # content, unless it's wrapped in a file-like object -- this was
        # the actual cause of "switching models" failing.
        window_df = pd.read_json(io.StringIO(payload['raw_features']))
        prediction = run_model_math(payload['model_name'], window_df)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
