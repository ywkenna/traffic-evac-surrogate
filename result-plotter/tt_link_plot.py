import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==========================================
# 논문용 폰트 설정 (8pt Times New Roman)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,              # 기본 글자 크기
    "axes.titlesize": 13,         # 제목 크기
    "axes.labelsize": 13,         # 축 이름 크기
    "xtick.labelsize": 13,        # x축 눈금 크기
    "ytick.labelsize": 13,        # y축 눈금 크기
    "legend.fontsize": 13,        # 범례 크기
    "figure.titlesize": 13        # 전체 제목 크기
})

# ==========================================
# 1. 경로 및 설정 (제공해주신 config 반영)
# ==========================================
BASE_PATH = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate")
DATASET_PATH = BASE_PATH / "dataset" / "dataset_final.csv"
MODEL_DIR = BASE_PATH / "models"  # config_surrogate.py의 MODEL_DIR과 동일하게 설정
PLOT_DIR = BASE_PATH / "plots(4y)"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 모델 구조 설정
HIDDEN_DIMS = [64, 128, 64]
DROPOUT = 0.4
RANDOM_SEED = 42
TEST_SIZE = 0.2

# 7개의 타겟 컬럼
Y_COLS = [
    "n_evac_arrived",
    "arrival_rate",
    #"tt_mean", "tt_median", "tt_p90",
    "links_mean", "links_p90"
]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def main():
    # 1. 파일 로드
    with open(MODEL_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(MODEL_DIR / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # [핵심 수정] 스케일러 크기 불일치(9개 vs 7개) 해결 로직
    if hasattr(y_scaler, 'mean_') and len(y_scaler.mean_) != len(Y_COLS):
        print(f"[알림] 스케일러 크기({len(y_scaler.mean_)})와 타겟 개수({len(Y_COLS)})가 다릅니다.")
        print("스케일러를 현재 타겟(앞의 7개)에 맞춰 조정합니다.")
        y_scaler.mean_ = y_scaler.mean_[:len(Y_COLS)]
        y_scaler.var_ = y_scaler.var_[:len(Y_COLS)]
        y_scaler.scale_ = y_scaler.scale_[:len(Y_COLS)]
        if hasattr(y_scaler, 'n_features_in_'):
            y_scaler.n_features_in_ = len(Y_COLS)

    # 2. 데이터 준비
    df = pd.read_csv(DATASET_PATH)
    X_cols = meta["x_cols"] 
    
    X = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Y_orig = df[Y_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    _, X_test, _, Y_test_orig = train_test_split(
        X, Y_orig, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # 3. 모델 로드 및 예측
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=len(X_cols), out_dim=len(Y_COLS),
                hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    
    model.load_state_dict(torch.load(MODEL_DIR / "surrogate_latest.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        X_test_s = x_scaler.transform(X_test).astype(np.float32)
        pred_s = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    # 4. 역변환
    pred_log = y_scaler.inverse_transform(pred_s)
    pred_final = np.expm1(pred_log)

    # 5. 산점도 생성
    for i, yname in enumerate(Y_COLS):
        y_true = Y_test_orig[:, i]
        y_pred = pred_final[:, i]
        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, y_pred, alpha=0.4, color='royalblue', edgecolors='white')
        
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Identity Line')
        
        plt.title(f"Target: {yname} (R² = {r2:.4f})", fontsize=13)
        plt.xlabel("Actual Simulation Value")
        plt.ylabel("Surrogate Prediction")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = PLOT_DIR / f"scatter_{yname}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path.name}")

if __name__ == "__main__":
    main()