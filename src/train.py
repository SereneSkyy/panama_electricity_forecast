import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_utils import load_and_clean_data, add_features, create_sequences
from architectures import RNNModel, LSTMModel
from paths import DATA_PATH, MODELS_DIR

# Settings
SEQ_LEN = 24
BATCH_SIZE = 64
EPOCHS_BY_MODEL = {"RNN": 20, "LSTM": 40}  # LSTM gets more epochs: it has 4x the gate
                                            # parameters of a plain RNN, so it needs more
                                            # training to reach a comparable fit (see report)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_engine(model, loader, criterion, optimizer):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb.unsqueeze(-1))
        loss.backward()
        optimizer.step()

def evaluate_engine(model, X_test, y_test, target_scaler):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).float().to(DEVICE)).cpu().numpy()

    preds_mw = target_scaler.inverse_transform(preds)
    actual_mw = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual_mw, preds_mw)
    rmse = np.sqrt(mean_squared_error(actual_mw, preds_mw))
    mape = np.mean(np.abs((actual_mw - preds_mw) / actual_mw)) * 100
    r2 = r2_score(actual_mw, preds_mw)
    return mae, rmse, mape, r2

# Start Processing
print("--- Loading and Engineering Data ---")
df = load_and_clean_data(DATA_PATH)
df_features = add_features(df)

# Split and Scale
train_size = int(len(df_features) * 0.8)
train_df = df_features[:train_size]
test_df = df_features[train_size:]

target_scaler = MinMaxScaler().fit(train_df[['nat_demand']])
feat_scaler = MinMaxScaler().fit(train_df.iloc[:, 1:])

# Save scalers for FastAPI
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(target_scaler, MODELS_DIR / "target_scaler.pkl")
joblib.dump(feat_scaler, MODELS_DIR / "feat_scaler.pkl")

# Prepare Tensors
def prepare_data(data_df):
    scaled_target = target_scaler.transform(data_df[['nat_demand']])
    scaled_feats = feat_scaler.transform(data_df.iloc[:, 1:])
    combined = np.hstack([scaled_target, scaled_feats])
    return create_sequences(combined, SEQ_LEN)

X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()),
    batch_size=BATCH_SIZE, shuffle=True
)

# COMPARE MODELS
results = {}

for name, model_class in [("RNN", RNNModel), ("LSTM", LSTMModel)]:
    epochs = EPOCHS_BY_MODEL[name]
    print(f"--- Training {name} ({epochs} epochs) ---")
    model = model_class(input_size=19, hidden_size=64, num_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        train_engine(model, train_loader, criterion, optimizer)
    
    mae, rmse, mape, r2 = evaluate_engine(model, X_test, y_test, target_scaler)
    results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    torch.save(model.state_dict(), MODELS_DIR / f"{name.lower()}_model.pth")

print("\n--- FINAL COMPARISON ---")
for m, metrics in results.items():
    print(f"{m} -> MAE: {metrics['MAE']:.2f} MW, RMSE: {metrics['RMSE']:.2f} MW, "
          f"MAPE: {metrics['MAPE']:.2f}%, R2: {metrics['R2']:.4f}")