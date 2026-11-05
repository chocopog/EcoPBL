"""PyTorch LSTM model architecture, training loop, and prediction."""

import numpy as np
import torch
import torch.nn as nn
from config import HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, LEARNING_RATE, PATIENCE, FEATURES, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X, y, window):
    """Create sliding-window input/target pairs."""
    xs, ys = [], []
    for i in range(window, len(X) + 1):
        xs.append(X[i - window:i])
        ys.append(y[i - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _batch_gen(X, y, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), BATCH_SIZE):
        j = idx[i:i + BATCH_SIZE]
        yield torch.tensor(X[j]).to(device), torch.tensor(y[j]).to(device)


def predict(model, X):
    """Run inference on a set of sequences."""
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in _batch_gen(X, np.zeros(len(X), dtype=np.float32), False):
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def train_lstm(X_tr, y_tr, X_v, y_v, epochs, target_scale):
    """Train LSTM and return (model, metrics_dict).

    *target_scale* is scaler_y.scale_[0], used to convert scaled RMSE
    back to percentage-return units for display.
    """
    model = LSTMModel(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss, best_weights, wait = np.inf, None, 0
    train_rmse_hist, val_rmse_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in _batch_gen(X_tr, y_tr, True):
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(loss.item())

        train_mse = float(np.mean(batch_losses))
        val_preds = predict(model, X_v)
        val_mse = float(np.mean((val_preds - y_v) ** 2))

        train_rmse_hist.append(round(float(np.sqrt(train_mse)) * target_scale * 100, 4))
        val_rmse_hist.append(round(float(np.sqrt(val_mse)) * target_scale * 100, 4))

        if val_mse < best_loss:
            best_loss, wait = val_mse, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    best_epoch = int(np.argmin(val_rmse_hist))
    return model, {
        "epochs_run": len(train_rmse_hist),
        "best_epoch": best_epoch + 1,
        "train_rmse": train_rmse_hist,
        "val_rmse": val_rmse_hist,
        "final_train_rmse": train_rmse_hist[-1],
        "best_val_rmse": val_rmse_hist[best_epoch],
    }

