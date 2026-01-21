import re
from pathlib import Path
import pandas as pd

DATASET_IN = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\dataset_with_dist.csv")
DATASET_OUT = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\dataset_final.csv")

NUM_ZONES = 32
NUM_SHELTERS = 6

def main():
    df = pd.read_csv(DATASET_IN)

    # 1) N=7000 고정 관련: state_N_total_* 전부 제거
    drop_cols = [c for c in df.columns if c.startswith("state_N_total_")]

    # 2) state_zone{z}_sum/mean/std/max/min 제거하고 alloc 하나만 남기기
    #    - 만약 현재 컬럼명이 state_zone14_sum 같은 형태라면 -> 제거
    zone_stat_pat = re.compile(r"^state_zone(\d+)_(sum|mean|std|max|min)$")
    drop_cols += [c for c in df.columns if zone_stat_pat.match(c)]

    #    - 그리고 alloc 컬럼을 만들어야 하는데,
    #      지금 데이터에 "state_zone{z}_alloc"이 이미 있으면 OK
    #      없다면, 기존 sum(혹은 mean) 값을 alloc으로 복사해서 생성
    for z in range(NUM_ZONES):
        alloc_col = f"state_zone{z}_alloc"
        if alloc_col not in df.columns:
            # sum 우선, 없으면 mean, 없으면 max 등에서 가져오기
            src_candidates = [
                f"state_zone{z}_sum",
                f"state_zone{z}_mean",
                f"state_zone{z}_max",
                f"state_zone{z}_min",
            ]
            src = next((c for c in src_candidates if c in df.columns), None)
            if src is None:
                # 정말 아무것도 없으면 0으로 채움(원하면 에러로 바꿔도 됨)
                df[alloc_col] = 0
            else:
                df[alloc_col] = df[src]

    # 3) alloc_shelter_*_veh vs assigned_to_shelter_* 중복 제거
    alloc_cols = [f"alloc_shelter_{i}_veh" for i in range(NUM_SHELTERS)]
    assigned_cols = [f"assigned_to_shelter_{i}" for i in range(NUM_SHELTERS)]

    has_alloc = any(c in df.columns for c in alloc_cols)
    has_assigned = any(c in df.columns for c in assigned_cols)

    # 둘 다 있으면 alloc 쪽 제거(추천)
    if has_alloc and has_assigned:
        drop_cols += [c for c in alloc_cols if c in df.columns]

    # drop 실행
    drop_cols = sorted(set([c for c in drop_cols if c in df.columns]))
    df = df.drop(columns=drop_cols)

    # 컬럼 정렬(원하면): state_id, policy_id 먼저
    front = [c for c in ["state_id", "policy_id"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_OUT, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved: {DATASET_OUT}")
    print(f"[INFO] columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
