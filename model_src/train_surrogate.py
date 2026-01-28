import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config_surrogate import (
    DATASET_CSV, MODEL_DIR,
    DROP_COLS, X_PREFIXES, Y_COLS,
    RANDOM_SEED, TEST_SIZE,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, HIDDEN_DIMS, DROPOUT
)

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

    X_PREFIXES = ["total_assigned", "sector_", "alloc_", "dist_"]

    for y in Y_COLS:
        if y not in df.columns:
            raise ValueError(f"Y 컬럼이 dataset에 없음: {y}")

    x_cols = pick_x_columns(df)
    
    # 데이터 준비
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Y_orig = df[Y_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    # [핵심 수정 1] Y 값에 로그 변환 적용 (log1p = ln(1+x))
    Y_log = np.log1p(Y_orig)

    # 데이터 분할 (로그 변환된 Y_log 사용)
    X_train, X_test, Y_train_log, Y_test_log = train_test_split(
        X, Y_log, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    # 평가를 위해 원본 스케일의 테스트 정답셋도 별도로 분할 (동일 시드)
    _, _, _, Y_test_orig = train_test_split(
        X, Y_orig, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = x_scaler.transform(X_test).astype(np.float32)

    # 로그 변환된 타겟에 대해 스케일링 수행
    Y_train_s = y_scaler.fit_transform(Y_train_log).astype(np.float32)
    Y_test_s  = y_scaler.transform(Y_test_log).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(Y_train_s))
    test_ds  = TensorDataset(torch.from_numpy(X_test_s),  torch.from_numpy(Y_test_s))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X_train_s.shape[1], out_dim=Y_train_s.shape[1],
                hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()

    best_test = float("inf")
    best_path = MODEL_DIR / "surrogate_latest.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_mse_list, train_mae_list = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse_loss = loss_fn_mse(pred, yb)
            opt.zero_grad()
            mse_loss.backward()
            opt.step()
            train_mse_list.append(mse_loss.item())
            train_mae_list.append(loss_fn_mae(pred, yb).item())

        model.eval()
        test_mse_list, test_mae_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                test_mse_list.append(loss_fn_mse(pred, yb).item())
                test_mae_list.append(loss_fn_mae(pred, yb).item())

        tr_mse, te_mse = np.mean(train_mse_list), np.mean(test_mse_list)
        tr_mae, te_mae = np.mean(train_mae_list), np.mean(test_mae_list)

        if te_mse < best_test:
            best_test = te_mse
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:4d}] MSE(tr/te): {tr_mse:.4f}/{te_mse:.4f} | MAE(tr/te): {tr_mae:.4f}/{te_mae:.4f}")

    # 평가 단계
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_test_s = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    # [핵심 수정 2] 역변환 로직
    # 1. 스케일러 역변환 (로그 스케일로 복구)
    pred_test_log = y_scaler.inverse_transform(pred_test_s)
    # 2. 로그 역변환 (expm1 = exp(x) - 1)
    pred_test = np.expm1(pred_test_log)

    actual = Y_test_orig # 실제 단위 값
    predicted = pred_test # 실제 단위 예측값

    print("\n=== 상세 테스트 결과 (Test Evaluation) ===")
    for i, yname in enumerate(Y_COLS):
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        
        r = rmse(y_pred, y_true)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            mape_str = f"{mape:.2f}%"
        except:
            mape_str = "계산 불가"

        print(f"[{yname}]")
        print(f"  - RMSE:     {r:.6f}")
        print(f"  - MAE:      {mae:.6f}")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - 오차율(MAPE):{mape_str}")
        print("-" * 30)

    # Baseline 비교 (원본 스케일 기준)
    print("\n=== Baseline (Mean Predictor) ===")
    Y_train_orig, _, _, _ = train_test_split(Y_orig, Y_orig, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    for i, yname in enumerate(Y_COLS):
        y_true = Y_test_orig[:, i]
        y_pred_base = np.full_like(y_true, Y_train_orig[:, i].mean())
        r2_base = r2_score(y_true, y_pred_base)
        rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))
        print(f"[{yname}] baseline R²={r2_base:.4f}, RMSE={rmse_base:.4f}")

    # === Baseline 출력 코드 바로 아래에 추가하세요 ===

    # 1. 스케일러 저장 (학습된 통계량을 나중에 그대로 쓰기 위함)
    with open(MODEL_DIR / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(MODEL_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    # 2. 메타데이터 저장 (컬럼명, 하이퍼파라미터 등 기록)
    meta = {
        "x_cols": x_cols,
        "y_cols": Y_COLS,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[최종 완료] 모델 및 스케일러가 {MODEL_DIR}에 모두 저장되었습니다.")

if __name__ == "__main__":
    main()