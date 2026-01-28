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

# ==========================================
# 1. 논문용 폰트 및 스타일 설정
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "figure.titlesize": 13,
    "savefig.dpi": 600,
    "pdf.fonttype": 42
})


# ==========================================
# 2. 경로 및 설정 (소요 시간 지표 전용)
# ==========================================
BASE_PATH = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate")
DATASET_PATH = BASE_PATH / "dataset" / "dataset_final.csv"
MODEL_DIR = BASE_PATH / "models"
PLOT_DIR = BASE_PATH / "plots_tt_analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 소요 시간 관련 3개 지표 정의
Y_COLS_TT = ["tt_mean", "tt_median", "tt_p90"]

STYLES_TT = {
    "tt_mean":   {"color": "#FF7F00", "marker": "o", "label": "Mean Travel Time"},
    "tt_median": {"color": "#4DAF4A", "marker": "s", "label": "Median Travel Time"},
    "tt_p90":    {"color": "#377EB8", "marker": "D", "label": "p90 Travel Time"}
}

# [MLP 구조는 이전과 동일]
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
    # 1. 메타 데이터 및 스케일러 로드
    with open(MODEL_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(MODEL_DIR / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # 2. 데이터 준비
    df = pd.read_csv(DATASET_PATH)
    X_cols = meta["x_cols"] 
    # 실제 데이터셋에서 TT 컬럼들을 포함한 Y 추출
    Y_orig = df[Y_COLS_TT].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    X = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    _, X_test, _, Y_test_orig = train_test_split(
        X, Y_orig, test_size=0.2, random_state=42
    )

    # 3. 모델 예측 (기존 7개 혹은 9개 출력 모델에서 TT 인덱스 추출 가정)
    # ※ 주의: 현재 로드하는 모델이 TT를 예측하도록 학습된 모델이어야 합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 실제 학습된 모델의 out_dim에 맞춰 설정 (예: 7개 타겟 모델)
    FULL_Y_COUNT = 7 
    model = MLP(in_dim=len(X_cols), out_dim=FULL_Y_COUNT, hidden_dims=[64, 128, 64], dropout=0.4).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "surrogate_latest.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        X_test_s = x_scaler.transform(X_test).astype(np.float32)
        pred_s = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()
    
    # 역변환 후 TT에 해당하는 인덱스만 슬라이싱 (예: 2, 3, 4번 인덱스가 TT라고 가정)
    pred_final_full = np.expm1(y_scaler.inverse_transform(pred_s))
    # Y_COLS_TT 순서에 맞는 인덱스로 추출하세요. (아래는 예시 인덱스)
    pred_final_tt = pred_final_full[:, [2, 3, 4]] 

    # 4. 시각화 (범례 분리 배치)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Identity Line (좌상단용)
    line_id, = ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Identity Line', zorder=1)

    handles_metrics = []
    for i, yname in enumerate(Y_COLS_TT):
        y_true = Y_test_orig[:, i]
        y_pred = pred_final_tt[:, i]
        
        # 정규화 (Min-Max)
        v_min, v_max = y_true.min(), y_true.max()
        y_true_norm = (y_true - v_min) / (v_max - v_min)
        y_pred_norm = (y_pred - v_min) / (v_max - v_min)
        
        r2 = r2_score(y_true, y_pred)
        style = STYLES_TT[yname]
        label_str = f"{style['label']} ($R^2$={r2:.3f})"
        
        sc = ax.scatter(y_true_norm, y_pred_norm, c=style['color'], marker=style['marker'], 
                        s=15, alpha=0.4, edgecolors='none', label=label_str, zorder=2)
        handles_metrics.append(sc)

    ax.set_xlabel("Normalized Actual Value (Time)", fontweight='bold')
    ax.set_ylabel("Normalized Predicted Value (Time)", fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=':', alpha=0.4)

    # 범례 1: Identity Line (좌상단)
    leg1 = ax.legend(handles=[line_id], loc='upper left', frameon=True, edgecolor='gray')
    ax.add_artist(leg1)
    
    # 범례 2: TT 지표들 (우하단 1열)
    ax.legend(handles=handles_metrics, loc='lower right', ncol=1, frameon=True, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "Figure5_TT_Reliability_Analysis.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()