import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 기존 설정 임포트 (config_surrogate.py가 있다고 가정)
from config_surrogate import (
    DATASET_CSV, MODEL_DIR,
    DROP_COLS, X_PREFIXES, Y_COLS,
    RANDOM_SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, HIDDEN_DIMS, DROPOUT
)

# 9:1 분할을 위한 설정 강제 지정
TEST_SIZE = 0.1 
N_SPLITS = 5 # 5-fold

def pick_x_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in DROP_COLS: continue
        if c in Y_COLS: continue
        if any(c.startswith(p) for p in X_PREFIXES):
            cols.append(c)
    return cols

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"dataset_final.csv not found: {DATASET_CSV}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)

    # 1. 32개 Zone -> 8개 Sector로 합산
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for i, sector_name in enumerate(sectors):
        zone_ids = [i + (j * 8) for j in range(4)]
        zone_cols = [f"state_zone{zid}_alloc" for zid in zone_ids]
        df[f"sector_{sector_name}_total"] = df[zone_cols].sum(axis=1)

    X_PREFIXES_UPDATED = ["total_assigned", "sector_", "alloc_", "dist_", "state_zone", "policy_"]
    x_cols = pick_x_columns(df)
    
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Y_orig = df[Y_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Y_log = np.log1p(Y_orig)

    # [수정] 1단계: 9:1 Hold-out 분할 (10%는 최종 테스트용)
    X_train_full, X_test, Y_train_full_log, Y_test_log = train_test_split(
        X, Y_log, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    _, _, _, Y_test_orig = train_test_split(
        X, Y_orig, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # [수정] 2단계: 5-fold Cross Validation 준비
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Starting {N_SPLITS}-fold Cross Validation (Hold-out Test Size: {TEST_SIZE}) ===")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n>> Fold {fold+1}/{N_SPLITS}")
        
        # 데이터 분리
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        Y_tr_log, Y_val_log = Y_train_full_log[train_idx], Y_train_full_log[val_idx]

        # 스케일링 (각 Fold마다 독립적으로 수행)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_tr_s = x_scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = x_scaler.transform(X_val).astype(np.float32)
        Y_tr_s = y_scaler.fit_transform(Y_tr_log).astype(np.float32)
        Y_val_s = y_scaler.transform(Y_val_log).astype(np.float32)

        train_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(Y_tr_s))
        val_ds = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(Y_val_s))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = MLP(in_dim=X_tr_s.shape[1], out_dim=Y_tr_s.shape[1],
                    hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn_mse = nn.MSELoss()

        best_val_mse = float("inf")
        fold_model_path = MODEL_DIR / f"model_fold_{fold+1}.pt"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn_mse(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    val_losses.append(loss_fn_mse(pred, yb).item())
            
            avg_val_mse = np.mean(val_losses)
            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                torch.save(model.state_dict(), fold_model_path)

        print(f"Fold {fold+1} Best Val MSE: {best_val_mse:.4f}")
        fold_results.append(best_val_mse)

    # 3. 최종 평가 (마지막 Fold의 스케일러와 모델 사용 예시)
    print("\n=== Final Hold-out Test Evaluation (9:1 Split) ===")
    # 실제로는 모든 Fold 모델의 앙상블을 쓰거나, 성능이 가장 좋은 Fold 모델을 선택합니다.
    # 여기서는 마지막 학습된 x_scaler, y_scaler와 해당 모델을 활용합니다.
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    model.load_state_dict(torch.load(MODEL_DIR / f"model_fold_{np.argmin(fold_results)+1}.pt"))
    model.eval()

    with torch.no_grad():
        pred_test_s = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    pred_test_log = y_scaler.inverse_transform(pred_test_s)
    pred_test = np.expm1(pred_test_log)

    for i, yname in enumerate(Y_COLS):
        y_true, y_pred = Y_test_orig[:, i], pred_test[:, i]
        print(f"[{yname}] RMSE: {rmse(y_pred, y_true):.4f}, R2: {r2_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()