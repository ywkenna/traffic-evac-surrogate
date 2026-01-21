import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from config_surrogate import MODEL_DIR

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_artifacts(model_dir: Path):
    with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(model_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(model_dir / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    model = MLP(
        in_dim=len(meta["x_cols"]),
        out_dim=len(meta["y_cols"]),
        hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"],
    )
    sd = torch.load(model_dir / "surrogate_latest.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return meta, x_scaler, y_scaler, model

def predict(df_in: pd.DataFrame) -> pd.DataFrame:
    meta, x_scaler, y_scaler, model = load_artifacts(MODEL_DIR)
    x_cols = meta["x_cols"]
    y_cols = meta["y_cols"]

    # 필요한 입력컬럼이 없으면 0으로 채움
    for c in x_cols:
        if c not in df_in.columns:
            df_in[c] = 0.0

    X = df_in[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    Xs = x_scaler.transform(X).astype(np.float32)

    with torch.no_grad():
        Ys = model(torch.from_numpy(Xs)).numpy()
    Y = y_scaler.inverse_transform(Ys)

    out = df_in.copy()
    for i, y in enumerate(y_cols):
        out[f"pred_{y}"] = Y[:, i]
    return out

if __name__ == "__main__":
    # 예시: 학습용 dataset_final.csv 일부를 읽어서 예측해보기
    sample_csv = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\dataset_final.csv")
    df = pd.read_csv(sample_csv).head(5)
    pred = predict(df)
    print(pred.filter(regex=r"^(state_id|policy_id|pred_)").to_string(index=False))
