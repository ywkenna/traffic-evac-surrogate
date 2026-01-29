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
# 1. 논문용 폰트 및 스타일 설정 (8pt Times New Roman)
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
# 2. 경로 및 설정
# ==========================================
BASE_PATH = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate")
DATASET_PATH = BASE_PATH / "dataset" / "dataset_final.csv"
MODEL_DIR = BASE_PATH / "models"
PLOT_DIR = BASE_PATH / "plots(4y_unified)" # 저장 폴더명 변경
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_DIMS = [64, 128, 64]
DROPOUT = 0.4
RANDOM_SEED = 42
TEST_SIZE = 0.2

Y_COLS = ["n_evac_arrived", "arrival_rate", "links_mean", "links_p90"]

# 시각화 스타일 설정 (색상, 마커, 표시 이름)
STYLES = {
    "n_evac_arrived": {"color": "#e41a1c", "marker": "o", "label": "Arrival Count"},
    "arrival_rate":   {"color": "#377eb8", "marker": "s", "label": "Arrival Rate"},
    "links_mean":     {"color": "#4daf4a", "marker": "^", "label": "Mean Path Length Proxy"},
    "links_p90":      {"color": "#984ea3", "marker": "D", "label": "p90 Path Length Proxy"}
}

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
    # 1. 파일 및 스케일러 로드
    with open(MODEL_DIR / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(MODEL_DIR / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    # 스케일러 크기 조정 (Y_COLS 개수에 맞춤)
    if hasattr(y_scaler, 'mean_') and len(y_scaler.mean_) != len(Y_COLS):
        y_scaler.mean_ = y_scaler.mean_[:len(Y_COLS)]
        y_scaler.var_ = y_scaler.var_[:len(Y_COLS)]
        y_scaler.scale_ = y_scaler.scale_[:len(Y_COLS)]

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

    # 로그 역변환
    pred_log = y_scaler.inverse_transform(pred_s)
    pred_final = np.expm1(pred_log)

    # 4. 통합 단일 산점도 생성 (범례 분리 로직 적용)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # [1] Identity Line (좌측 상단으로 보낼 범례)
    line1, = ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Identity Line', zorder=1)

    handles_metrics = [] # 지표 범례 핸들을 담을 리스트
    
    for i, yname in enumerate(Y_COLS):
        y_true = Y_test_orig[:, i]
        y_pred = pred_final[:, i]
        
        # [정규화] 0~1 Min-Max Scaling
        v_min, v_max = y_true.min(), y_true.max()
        y_true_norm = (y_true - v_min) / (v_max - v_min)
        y_pred_norm = (y_pred - v_min) / (v_max - v_min)
        
        r2 = r2_score(y_true, y_pred)
        style = STYLES[yname]
        label_str = f"{style['label']}"# ($R^2$={r2:.3f})"
        
        # 산점도 그리고 핸들 저장
        scatter = ax.scatter(y_true_norm, y_pred_norm, 
                             c=style['color'], marker=style['marker'], 
                             s=15, alpha=0.4, edgecolors='none', 
                             label=label_str, zorder=2)
        handles_metrics.append(scatter)

    # 축 설정
    ax.set_xlabel("Normalized Actual Value", fontweight='bold')
    ax.set_ylabel("Normalized Predicted Value", fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=':', alpha=0.4)

    # -----------------------------------------------------------
    # [핵심 수정] 범례 이중 배치 로직
    # -----------------------------------------------------------
    
    # 1. Identity Line 범례 (좌측 상단 - 단일 항목)
    leg1 = ax.legend(handles=[line1], loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')
    
    # 2. 첫 번째 범례가 사라지지 않도록 아티스트로 추가
    ax.add_artist(leg1)
    
    # 3. 지표 범례 (우측 하단 - 2열 배치)
    ax.legend(
        handles=handles_metrics,
        loc='lower right', 
        ncol=2, 
        frameon=True, 
        framealpha=0.9, 
        edgecolor='gray',
        columnspacing=0.8,
        handletextpad=0.1
    )

    plt.tight_layout()
    save_path = PLOT_DIR / "Figure4_Final_Split_Legend.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"범례가 분리된 그래프가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()