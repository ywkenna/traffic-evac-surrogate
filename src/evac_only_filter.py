from pathlib import Path
import pandas as pd

# ============================================================
# 1. 경로 설정 (여기만 필요에 따라 수정)
# ============================================================

# trajectory CSV들이 들어있는 디렉토리
INPUT_DIR = Path("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/sim-sorted")  
# 예: Path("/home/etri33533/data/trajectory")

# 결과를 저장할 디렉토리
OUTPUT_DIR = Path("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/sim-evac-only")

# 대피소 edge 정보 엑셀
EVAC_AREA_XLSX = Path("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/data/evac-area.xlsx")

# trajectory 파일 패턴
FILE_PATTERN = "sorted_2026shelterseek_12_*_trajectory.csv"

# evac-area.xlsx 안의 대피소 edge 컬럼명
EVAC_EDGE_COL = "edge_id"


# ============================================================
# 2. evac-area.xlsx 로딩
# ============================================================

def load_evac_edges(xlsx_path: Path, edge_col: str) -> set:
    df = pd.read_excel(xlsx_path)

    if edge_col not in df.columns:
        raise ValueError(
            f"[ERROR] '{edge_col}' column not found in {xlsx_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    # 문자열 통일 (ID 타입 꼬임 방지)
    return set(df[edge_col].dropna().astype(str).unique())


# ============================================================
# 3. 단일 파일 처리
# ============================================================

def filter_one_trajectory(csv_path: Path, evac_edges: set):
    df = pd.read_csv(csv_path)

    required_cols = {"vehID", "seq", "linkID"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] {csv_path.name} missing columns: {missing}")

    df["vehID"] = df["vehID"].astype(str)
    df["linkID"] = df["linkID"].astype(str)

    # 차량별 마지막 기록 (최대 seq)
    last_idx = df.groupby("vehID")["seq"].idxmax()
    last_rows = df.loc[last_idx, ["vehID", "linkID"]]

    # 최종 목적지가 대피소 edge 인 차량
    evac_veh_ids = set(
        last_rows[last_rows["linkID"].isin(evac_edges)]["vehID"]
    )

    # 해당 차량들의 모든 trajectory 기록 유지
    evac_df = df[df["vehID"].isin(evac_veh_ids)].copy()

    return evac_df, len(df), len(evac_df)


# ============================================================
# 4. 전체 파일 일괄 처리
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"[ERROR] INPUT_DIR not found: {INPUT_DIR}")
    if not EVAC_AREA_XLSX.exists():
        raise FileNotFoundError(f"[ERROR] EVAC_AREA_XLSX not found: {EVAC_AREA_XLSX}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evac_edges = load_evac_edges(EVAC_AREA_XLSX, EVAC_EDGE_COL)

    traj_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

    if not traj_files:
        print(f"[WARN] No files matched: {FILE_PATTERN}")
        return

    for traj_file in traj_files:
        evac_df, total_rows, kept_rows = filter_one_trajectory(
            traj_file, evac_edges
        )

        out_name = f"evac-only_{traj_file.name}"
        out_path = OUTPUT_DIR / out_name
        evac_df.to_csv(out_path, index=False)

        print(
            f"[OK] {traj_file.name} → {out_name} "
            f"| rows: {kept_rows}/{total_rows}"
        )

    print(f"\n[DONE] evac-only files written to: {OUTPUT_DIR.resolve()}")


# ============================================================
# 5. 실행
# ============================================================

if __name__ == "__main__":
    main()
