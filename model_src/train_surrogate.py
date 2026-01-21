import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

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
        if c in DROP_COLS:
            continue
        if c in Y_COLS:
            continue
        # prefix 기반 선택
        if any(c.startswith(p) for p in X_PREFIXES):
            cols.append(c)
    # state_id/policy_id를 쓰고 싶으면 여기에서 추가 가능
    # cols += [c for c in ["state_id", "policy_id"] if c in df.columns]
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

    # 출력(Y) 존재 체크
    for y in Y_COLS:
        if y not in df.columns:
            raise ValueError(f"Y 컬럼이 dataset에 없음: {y}")

    x_cols = pick_x_columns(df)
    if len(x_cols) == 0:
        raise ValueError("X 컬럼을 하나도 못 골랐음. X_PREFIXES / DROP_COLS 확인 필요")

    # 숫자화(혹시 object 섞이면 NaN 처리)
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Y = df[Y_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = x_scaler.transform(X_test).astype(np.float32)

    Y_train_s = y_scaler.fit_transform(Y_train).astype(np.float32)
    Y_test_s  = y_scaler.transform(Y_test).astype(np.float32)

    # torch dataset
    train_ds = TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(Y_train_s))
    test_ds  = TensorDataset(torch.from_numpy(X_test_s),  torch.from_numpy(Y_test_s))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X_train_s.shape[1], out_dim=Y_train_s.shape[1],
                hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_test = float("inf")
    best_path = MODEL_DIR / "surrogate_latest.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                test_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        te = float(np.mean(test_losses))

        if te < best_test:
            best_test = te
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:4d}] train={tr:.5f} test={te:.5f} best={best_test:.5f}")

    # best 모델 로드 후 실제 스케일로 평가
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_test_s = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    pred_test = y_scaler.inverse_transform(pred_test_s)
    y_test    = Y_test

    # 간단 지표 출력
    print("\n=== Test RMSE (per target) ===")
    for i, yname in enumerate(Y_COLS):
        r = rmse(pred_test[:, i], y_test[:, i])
        print(f"{yname:20s} RMSE = {r:.6f}")

    # scaler 저장
    with open(MODEL_DIR / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(MODEL_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    meta = {
        "x_cols": x_cols,
        "y_cols": Y_COLS,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "test_size": TEST_SIZE,
        "seed": RANDOM_SEED,
    }
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        

    # 기존 코드에서 저장했던 값을 다시 한번 확인합니다.
    actual = Y_test  
    predicted = pred_test

    print("\n=== 상세 테스트 결과 (Test Evaluation) ===")
    for i, yname in enumerate(Y_COLS):
        # i번째 컬럼의 실제값과 예측값 추출
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        
        # 1. RMSE (기존 지표)
        r = rmse(y_pred, y_true)
        
        # 2. R² Score (상대적 정확도: 1에 가까울수록 완벽)
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        r2 = r2_score(y_true, y_pred)
        
        # 3. MAPE (평균 오차율: % 단위, 낮을수록 좋음)
        try:
            # 실제값이 0이면 MAPE는 무한대가 될 수 있어 처리
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            mape_str = f"{mape:.2f}%"
        except:
            mape_str = "계산 불가"

        print(f"[{yname}]")
        print(f"  - RMSE:     {r:.6f}")
        print(f"  - R² Score: {r2:.4f} (1에 가까울수록 우수)")
        print(f"  - 오차율:    {mape_str} (낮을수록 우수)")
        print("-" * 30)
    # === 여기까지가 main() 함수 내부입니다 ===

    
    print(f"\n[DONE] saved model/scalers/meta to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
