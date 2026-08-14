# Panama Energy AI — RNN vs. LSTM vs. Ensemble Demand Forecasting

## Structure

- **`Panama_Forecasting.ipynb`** — the whole pipeline in one notebook: load data, clean, EDA,
  feature engineering, train RNN and LSTM (6-hour multi-step forecast), evaluate both plus a
  simple average ensemble, and save everything `main.py` needs. This replaces the old `src/`
  folder entirely — there's nothing else to look through.
- **`main.py`** — self-contained FastAPI backend.;
  the model classes and data-prep functions are defined directly in this file, matching the
  notebook exactly.
- **`dashboard/`** — plain HTML/CSS/JS frontend, no build step, no external dependencies (the
  chart is hand-drawn SVG, not a CDN library).
- **`models/`** — `target_scaler.pkl`, `feat_scaler.pkl`, `rnn_model.pth`, `lstm_model.pth`,
  produced by the notebook's last cell.

## What changed in this pass

1. **Reproducibility**: `set_seed(42)` is called before training each model, so results are
   deterministic run to run (previously unseeded — the same LSTM config gave MAE 72.6 vs. 76.1 on
   two different runs).
2. **Multi-step forecasting**: models now predict 6 hours ahead (`PRED_LEN = 6`) instead of just
   1 hour.
3. **Ensemble**: a simple average of the RNN and LSTM predictions, evaluated alongside both
   individual models in the notebook's final comparison table.

The 80/20 chronological train/test split (no validation set)

## Running it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open and run Panama_Forecasting.ipynb top to bottom
#    -> writes models/*.pth and models/*.pkl

# 3. Start the backend
uvicorn main:app --reload
# -> running at http://127.0.0.1:8000

# 4. Open the dashboard
python -m http.server 5500 --directory dashboard
# -> open http://127.0.0.1:5500
```

