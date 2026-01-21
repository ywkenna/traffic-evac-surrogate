from pathlib import Path
import pandas as pd

# ====== 경로 하드코딩 ======
DATASET_IN = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\processed\dataset.csv")
CASE_CSV   = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\data\case1000_zone_shelter_alloc_cap2000.csv")
DIST_CSV   = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\data\shelter_zone_dist.csv")

OUT_CSV    = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\dataset_with_dist.csv")

NUM_SHELTERS = 6


def main():
    for p in [DATASET_IN, CASE_CSV, DIST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}")

    base = pd.read_csv(DATASET_IN)
    cases = pd.read_csv(CASE_CSV)
    dist = pd.read_csv(DIST_CSV)

    # --- dist 컬럼 체크 ---
    need_dist_cols = ["zone32_id"] + [f"shelter{i}" for i in range(NUM_SHELTERS)]
    missing = [c for c in need_dist_cols if c not in dist.columns]
    if missing:
        raise ValueError(f"shelter_zone_dist.csv 컬럼이 예상과 다름. 누락: {missing}, 현재: {list(dist.columns)}")

    # --- cases 컬럼 체크 ---
    need_case_cols = ["state_id", "policy_id", "zone32_id", "shelter_id", "assigned_vehicles"]
    missing = [c for c in need_case_cols if c not in cases.columns]
    if missing:
        raise ValueError(f"case csv 컬럼이 예상과 다름. 누락: {missing}, 현재: {list(cases.columns)}")

    # 타입 정리
    cases["state_id"] = cases["state_id"].astype(int)
    cases["policy_id"] = cases["policy_id"].astype(int)
    cases["zone32_id"] = cases["zone32_id"].astype(int)
    cases["shelter_id"] = cases["shelter_id"].astype(int)
    cases["assigned_vehicles"] = cases["assigned_vehicles"].astype(float)

    dist["zone32_id"] = dist["zone32_id"].astype(int)
    for i in range(NUM_SHELTERS):
        dist[f"shelter{i}"] = dist[f"shelter{i}"].astype(float)

    # zone32_id 기준으로 거리 붙이기
    merged = cases.merge(dist, on="zone32_id", how="left")
    if merged[[f"shelter{i}" for i in range(NUM_SHELTERS)]].isna().any().any():
        bad = merged[merged[[f"shelter{i}" for i in range(NUM_SHELTERS)]].isna().any(axis=1)].head(10)
        raise ValueError(f"거리 매칭 실패(zone32_id 불일치). 예시:\n{bad[['state_id','policy_id','zone32_id']]}")

    # --- 케이스별 feature 계산 ---
    # 1) 전체 가중 평균 거리: sum(assigned_vehicles * dist(zone, assigned_shelter)) / sum(assigned_vehicles)
    merged["dist_assigned"] = merged.apply(
        lambda r: r[f"shelter{int(r['shelter_id'])}"], axis=1
    )
    merged["w_dist_assigned"] = merged["assigned_vehicles"] * merged["dist_assigned"]

    overall = (
        merged.groupby(["state_id", "policy_id"], as_index=False)
        .agg(total_assigned=("assigned_vehicles", "sum"),
             sum_wdist=("w_dist_assigned", "sum"))
    )
    overall["dist_to_assigned_shelter_mean"] = overall["sum_wdist"] / overall["total_assigned"]
    overall = overall.drop(columns=["sum_wdist"])

    # 2) shelter별 가중 평균 거리(“그 shelter로 배정된” 차량들 기준)
    per_shelter_rows = []
    for s in range(NUM_SHELTERS):
        m = merged[merged["shelter_id"] == s].copy()
        if len(m) == 0:
            continue
        m["w"] = m["assigned_vehicles"] * m[f"shelter{s}"]
        g = (
            m.groupby(["state_id", "policy_id"], as_index=False)
             .agg(v=("assigned_vehicles", "sum"), w=("w", "sum"))
        )
        g[f"dist_to_shelter{s}_mean"] = g["w"] / g["v"]
        g = g.drop(columns=["v", "w"])
        per_shelter_rows.append(g)

    if per_shelter_rows:
        per_shelter = per_shelter_rows[0]
        for g in per_shelter_rows[1:]:
            per_shelter = per_shelter.merge(g, on=["state_id", "policy_id"], how="outer")
    else:
        per_shelter = overall[["state_id", "policy_id"]].copy()

    feats = overall.merge(per_shelter, on=["state_id", "policy_id"], how="left")

    # base(dataset.csv)에 붙이기
    out = base.merge(feats, on=["state_id", "policy_id"], how="left")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    added = [c for c in out.columns if c.startswith("dist_")]
    print(f"[DONE] saved: {OUT_CSV}")
    print(f"[INFO] added dist cols: {added}")


if __name__ == "__main__":
    main()
