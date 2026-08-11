from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import io
from pathlib import Path

# Anchor paths to this file's own location, not the current working
# directory -- so the app works no matter what folder it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_SIZE = 19
HIDDEN_SIZE = 64
NUM_LAYERS = 2
PRED_LEN = 6   # model forecasts 6 hours ahead; the dashboard shows only the first (+1h)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS, DF_FULL, TARGET_SCALER, FEAT_SCALER = {}, None, None, None


class SpecificPredictionInput(BaseModel):
    data: list
    model_name: str


# ---------- Model architectures (same shape as the training notebook) ----------

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------- Data loading / feature engineering (same as the training notebook) ----------

def load_and_clean_data(path):
    df = pd.read_csv(path, parse_dates=["datetime"], dayfirst=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime").asfreq("h")
    df["nat_demand"] = df["nat_demand"].interpolate(method="linear")
    return df.reset_index()


def add_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    feature_cols = ["nat_demand", "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
                     "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
                     "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
                     "holiday", "school", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    return df[feature_cols]


# ---------- Startup: load trained models + scalers saved by the notebook ----------

@app.on_event("startup")
def startup_event():
    global DF_FULL, TARGET_SCALER, FEAT_SCALER, MODELS
    try:
        TARGET_SCALER = joblib.load(MODELS_DIR / "target_scaler.pkl")
        FEAT_SCALER = joblib.load(MODELS_DIR / "feat_scaler.pkl")
        for m_type, m_class in [("LSTM", LSTMModel), ("RNN", RNNModel)]:
            m = m_class(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                        num_layers=NUM_LAYERS, output_size=PRED_LEN)
            m.load_state_dict(torch.load(MODELS_DIR / f"{m_type.lower()}_model.pth", map_location="cpu"))
            m.eval()
            MODELS[m_type] = m

        raw_df = load_and_clean_data(DATA_PATH)
        ts = raw_df["datetime"].dt.strftime("%b %d, %Y - %H:%M").tolist()
        eng_df = add_features(raw_df)
        eng_df["datetime_str"] = ts
        DF_FULL = eng_df
        print("Backend ready — models and scalers loaded from models/.")
    except Exception as e:
        print(f"Startup error: {e}")


def run_model_math(model_name, window_df):
    target_df = window_df[["nat_demand"]]
    scaled_target = TARGET_SCALER.transform(target_df)
    feat_df = window_df.drop(columns=["nat_demand", "datetime_str"])
    scaled_feats = FEAT_SCALER.transform(feat_df)
    combined = np.hstack([scaled_target, scaled_feats])
    input_tensor = torch.tensor(combined).float().unsqueeze(0)

    model = MODELS.get(model_name.upper(), MODELS["LSTM"])
    with torch.no_grad():
        pred_scaled = model(input_tensor)   # shape (1, PRED_LEN)

    res_mw = TARGET_SCALER.inverse_transform(pred_scaled.numpy().reshape(-1, 1))
    return res_mw.flatten().tolist()   # [+1h, +2h, ..., +6h], all PRED_LEN hours


@app.get("/predict_sample/{model_name}")
async def predict_sample(model_name: str):
    try:
        idx = np.random.randint(int(len(DF_FULL) * 0.8), len(DF_FULL) - 25)
        window_df = DF_FULL.iloc[idx: idx + 24].copy()
        horizon = run_model_math(model_name, window_df)   # 6-hour forecast

        return {
            "history": window_df["nat_demand"].tolist(),
            "prediction": horizon[0],   # kept for backward compatibility (next-hour number)
            "horizon": horizon,         # full 6-hour forecast, +1h through +6h
            "timestamp": window_df.iloc[-1]["datetime_str"],
            "raw_features": window_df.to_json(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_specific")
async def predict_specific(payload: dict):
    try:
        # pd.read_json() treats a bare string as a file PATH, not JSON
        # content, unless wrapped in a file-like object.
        window_df = pd.read_json(io.StringIO(payload['raw_features']))
        horizon = run_model_math(payload['model_name'], window_df)
        return {"prediction": horizon[0], "horizon": horizon}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve dashboard/ directly -- this must be mounted AFTER the routes above,
# so /predict_sample and /predict_specific still match first. Anything else
# (/, /style.css, /app.js) falls through to the dashboard files, and
# html=True means "/" serves dashboard/index.html automatically.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "dashboard"), html=True), name="dashboard")
