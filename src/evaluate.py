import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), "src"))
from architectures import RNNModel, LSTMModel
from data_utils import load_and_clean_data, add_features, create_sequences
from paths import DATA_PATH, MODELS_DIR

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_metrics(model, X_test, y_test, target_scaler):
    # 1. SET TO EVALUATION MODE
    model.eval()
    
    # 2. DISABLE GRADIENTS FOR TESTING
    with torch.no_grad():
        X_tensor = torch.tensor(X_test).float()
        preds_scaled = model(X_tensor).cpu().numpy()
    
    # 3. INVERSE SCALE TO GET ACTUAL MW
    preds_mw = target_scaler.inverse_transform(preds_scaled).flatten()
    actual_mw = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 4. CALCULATE MATH
    mae = mean_absolute_error(actual_mw, preds_mw)
    rmse = np.sqrt(mean_squared_error(actual_mw, preds_mw))
    mape_val = calculate_mape(actual_mw, preds_mw)
    r2 = r2_score(actual_mw, preds_mw)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape_val, "R2": r2}

def run_full_evaluation():
    print("--- Starting Deep Learning Model Evaluation ---")
    
    # Load Data & Scalers
    df = load_and_clean_data(DATA_PATH)
    df_features = add_features(df)
    target_scaler = joblib.load(MODELS_DIR / "target_scaler.pkl")
    feat_scaler = joblib.load(MODELS_DIR / "feat_scaler.pkl")
    
    # Prepare Test Set (Same split logic as train.py)
    train_size = int(len(df_features) * 0.8)
    test_df = df_features[train_size:]
    
    scaled_target = target_scaler.transform(test_df[['nat_demand']])
    scaled_feats = feat_scaler.transform(test_df.iloc[:, 1:])
    combined = np.hstack([scaled_target, scaled_feats])
    X_test, y_test = create_sequences(combined, 24)

    # Comparison Loop
    comparison = []
    
    for name, model_class in [("RNN", RNNModel), ("LSTM", LSTMModel)]:
        # Instantiate and Load Weights
        model = model_class(input_size=19, hidden_size=64, num_layers=2)
        model_path = MODELS_DIR / f"{name.lower()}_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            metrics = get_metrics(model, X_test, y_test, target_scaler)
            metrics["Model"] = name
            comparison.append(metrics)
            print(f"{name} Evaluation Complete")
        else:
            print(f"Could not find {model_path}. Run train.py first.")

    # Display results as a nice table
    if comparison:
        results_df = pd.DataFrame(comparison).set_index("Model")
        print("\n" + "="*50)
        print("           FINAL PERFORMANCE COMPARISON")
        print("="*50)
        print(results_df.round(4).to_string())
        print("="*50)

if __name__ == "__main__":
    run_full_evaluation()