# build_dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 기본 경로 (레포 구조 기준)
# traffic-evac-surrogate/
#   data/
#   sim-outputs/
#   sim-evac-only/
#   sim-sorted/
#   src/
# =========================
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

DATA_DIR = REPO_ROOT / "data"
DEFAULT_COLLECTED = REPO_ROOT / "dataset" / "collected_metrics.csv"   # 너는 data에 두는 걸 추천
DEFAULT_STATE = DATA_DIR / "state25_zone32_N7000_pmz0_5km.csv"
DEFAULT_POLICY = DATA_DIR / "policy40_shelter_ratio.csv"
DEFAULT_CASE = DATA_DIR / "case1000_zone_shelter_alloc_cap2000.csv"

OUT_DIR = REPO_ROOT / "dataset" / "processed"
OUT_DATASET = OUT_DIR / "dataset.csv"


# =========================
# 유틸
# =========================
def _numeric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _safe_entropy(counts: np.ndarray) -> float:
    """counts -> Shannon entropy (natural log). counts can be float/int."""
    s = float(np.sum(counts))
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _safe_gini(x: np.ndarray) -> float:
    """Gini coefficient for nonnegative array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if np.min(x) < 0:
        # shift to nonnegative
        x = x - np.min(x)
    s = x.sum()
    if s == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    # Gini = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) / (n * s)) - (n + 1) / n)


# =========================
# 1) State features
# =========================
def build_state_features(state_csv: Path) -> pd.DataFrame:
    """
    state25_zone32_N7000_pmz0_5km.csv 가
    - (state_id, zone32_id) 단위로 여러 행일 가능성이 높아서
    state_id 단위로 숫자 컬럼을 집계(sum/mean/std/max/min)해서 feature를 만든다.
    """
    df = pd.read_csv(state_csv)
    if "state_id" not in df.columns:
        raise ValueError(f"[state csv] 'state_id' 컬럼이 없음: {state_csv}")

    df["state_id"] = df["state_id"].astype(int)

    exclude = {"state_id", "zone32_id", "zone_id", "zone"}
    num_cols = _numeric_cols(df, exclude=exclude)

    if not num_cols:
        # 숫자 컬럼이 거의 없을 경우를 대비
        out = df[["state_id"]].drop_duplicates().copy()
        out["state_row_count"] = df.groupby("state_id").size().values
        return out

    agg = {}
    for c in num_cols:
        agg[c] = ["sum", "mean", "std", "max", "min"]

    g = df.groupby("state_id", as_index=False).agg(agg)
    # multiindex columns flatten
    g.columns = ["state_id"] + [f"state_{c}_{stat}" for c, stat in g.columns[1:]]

    # std NaN -> 0
    std_cols = [c for c in g.columns if c.endswith("_std")]
    for c in std_cols:
        g[c] = g[c].fillna(0.0)

    return g


# =========================
# 2) Policy features
# =========================
def build_policy_features(policy_csv: Path) -> pd.DataFrame:
    """
    policy40_shelter_ratio.csv 형태가 다양할 수 있어서 두 케이스를 지원:
    A) policy_id + shelter_id + ratio(또는 weight) 형태 -> pivot 해서 shelter별 ratio feature 생성
    B) policy_id + 여러 숫자컬럼(이미 wide) -> 그대로 사용
    """
    df = pd.read_csv(policy_csv)
    if "policy_id" not in df.columns:
        raise ValueError(f"[policy csv] 'policy_id' 컬럼이 없음: {policy_csv}")
    df["policy_id"] = df["policy_id"].astype(int)

    # long format이면 pivot
    if "shelter_id" in df.columns:
        # ratio 컬럼 추정
        ratio_col = None
        for cand in ["ratio", "shelter_ratio", "weight", "value"]:
            if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
                ratio_col = cand
                break
        if ratio_col is None:
            # shelter_id는 있는데 ratio가 없으면 숫자컬럼 중 하나 선택
            num_cols = _numeric_cols(df, exclude={"policy_id", "shelter_id"})
            if not num_cols:
                raise ValueError(f"[policy csv] shelter_id는 있는데 ratio로 쓸 숫자 컬럼이 없음: {policy_csv}")
            ratio_col = num_cols[0]

        df["shelter_id"] = df["shelter_id"].astype(int)

        piv = df.pivot_table(
            index="policy_id",
            columns="shelter_id",
            values=ratio_col,
            aggfunc="mean"
        ).reset_index()

        # 컬럼명 정리
        new_cols = ["policy_id"] + [f"policy_shelter_ratio_{int(c)}" for c in piv.columns[1:]]
        piv.columns = new_cols

        # 요약통계(엔트로피/최대비중)
        ratio_mat = piv.drop(columns=["policy_id"]).to_numpy(dtype=float)
        ent = []
        maxshare = []
        for r in ratio_mat:
            r = np.nan_to_num(r, nan=0.0)
            s = r.sum()
            if s <= 0:
                ent.append(0.0)
                maxshare.append(0.0)
            else:
                ent.append(_safe_entropy(r))
                maxshare.append(float(np.max(r) / s))
        piv["policy_ratio_entropy"] = ent
        piv["policy_ratio_max_share"] = maxshare

        return piv

    # wide format: 숫자컬럼만 policy_ prefix 붙여서 사용
    exclude = {"policy_id"}
    num_cols = _numeric_cols(df, exclude=exclude)
    out = df[["policy_id"]].drop_duplicates().merge(df, on="policy_id", how="left")
    # 동일 policy_id 여러 줄이면 첫 줄만
    out = out.groupby("policy_id", as_index=False).first()

    rename_map = {}
    for c in out.columns:
        if c == "policy_id":
            continue
        rename_map[c] = f"policy_{c}"
    out = out.rename(columns=rename_map)

    return out


# =========================
# 3) Allocation features (case csv)
# =========================
def build_alloc_features(case_csv: Path, num_shelters: int = 6) -> pd.DataFrame:
    """
    case1000_zone_shelter_alloc_cap2000.csv에서
    (state_id, policy_id) 단위로 feature 생성

    - shelter별 총 배정 대수: alloc_shelter_0_veh ... alloc_shelter_5_veh
    - shelter 분포 요약: entropy, gini, max_share
    - zone별 분포 요약(선택): zone 배정 표준편차/최대/최소 등 (너무 wide로 32개 안 뽑고 요약만)
    """
    df = pd.read_csv(case_csv)

    required = {"state_id", "policy_id", "zone32_id", "shelter_id", "assigned_vehicles"}
    if not required.issubset(df.columns):
        raise ValueError(f"[case csv] 필요한 컬럼 {required} 중 일부가 없음. 현재={list(df.columns)}")

    df["state_id"] = df["state_id"].astype(int)
    df["policy_id"] = df["policy_id"].astype(int)
    df["zone32_id"] = df["zone32_id"].astype(int)
    df["shelter_id"] = df["shelter_id"].astype(int)
    df["assigned_vehicles"] = pd.to_numeric(df["assigned_vehicles"], errors="coerce").fillna(0).astype(int)

    # (state,policy,shelter) 합계 -> pivot
    sh = (
        df.groupby(["state_id", "policy_id", "shelter_id"], as_index=False)["assigned_vehicles"]
        .sum()
    )
    piv = sh.pivot_table(
        index=["state_id", "policy_id"],
        columns="shelter_id",
        values="assigned_vehicles",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # shelter 0~num_shelters-1 모두 컬럼이 없을 수 있으니 보정
    for sid in range(num_shelters):
        if sid not in piv.columns:
            piv[sid] = 0

    # 정렬 + 컬럼명 변경
    keep_cols = ["state_id", "policy_id"] + list(range(num_shelters))
    piv = piv[keep_cols]
    piv = piv.rename(columns={sid: f"alloc_shelter_{sid}_veh" for sid in range(num_shelters)})

    # 요약통계
    mat = piv[[f"alloc_shelter_{sid}_veh" for sid in range(num_shelters)]].to_numpy(dtype=float)
    ent, gini, maxshare = [], [], []
    for r in mat:
        r = np.nan_to_num(r, nan=0.0)
        s = r.sum()
        ent.append(_safe_entropy(r))
        gini.append(_safe_gini(r))
        maxshare.append(0.0 if s <= 0 else float(np.max(r) / s))
    piv["alloc_shelter_entropy"] = ent
    piv["alloc_shelter_gini"] = gini
    piv["alloc_shelter_max_share"] = maxshare
    piv["alloc_total_assigned"] = mat.sum(axis=1)

    # zone 분포 요약(32개를 wide로 안 뽑고, 요약만)
    z = (
        df.groupby(["state_id", "policy_id", "zone32_id"], as_index=False)["assigned_vehicles"]
        .sum()
    )
    zsum = z.groupby(["state_id", "policy_id"])["assigned_vehicles"].agg(
        alloc_zone_mean="mean",
        alloc_zone_std="std",
        alloc_zone_max="max",
        alloc_zone_min="min",
    ).reset_index()
    zsum["alloc_zone_std"] = zsum["alloc_zone_std"].fillna(0.0)

    out = piv.merge(zsum, on=["state_id", "policy_id"], how="left")
    return out


# =========================
# MAIN
# =========================
def main():
    # ---- 입력 파일들(필요하면 여기만 고쳐도 됨)
    collected_path = DEFAULT_COLLECTED
    state_path = DEFAULT_STATE
    policy_path = DEFAULT_POLICY
    case_path = DEFAULT_CASE

    # ---- load collected metrics (y)
    y = pd.read_csv(collected_path)
    for c in ["state_id", "policy_id"]:
        if c not in y.columns:
            raise ValueError(f"[collected_metrics] '{c}' 컬럼이 없음: {collected_path}")
    y["state_id"] = y["state_id"].astype(int)
    y["policy_id"] = y["policy_id"].astype(int)

    # ---- build X parts
    x_state = build_state_features(state_path)
    x_policy = build_policy_features(policy_path)
    x_alloc = build_alloc_features(case_path, num_shelters=6)

    # ---- merge
    ds = (
        y.merge(x_state, on="state_id", how="left")
         .merge(x_policy, on="policy_id", how="left")
         .merge(x_alloc, on=["state_id", "policy_id"], how="left")
    )

    # ---- 정렬 (요구사항)
    ds = ds.sort_values(["state_id", "policy_id"]).reset_index(drop=True)

    # ---- sanity: 누락 체크(있어도 저장은 하되, 경고만)
    missing_state = ds["state_id"].isna().sum()
    missing_policy = ds["policy_id"].isna().sum()
    # (위는 사실상 0일 텐데, 병합 실패 시 feature NaN이 생길 수 있음)
    n_feat_nan = int(ds.isna().sum().sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")

    print("[DONE] dataset saved:", OUT_DATASET)
    print("[INFO] rows:", len(ds), " cols:", len(ds.columns))
    print("[INFO] total NaN cells:", n_feat_nan)
    print("[INFO] head:\n", ds.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
