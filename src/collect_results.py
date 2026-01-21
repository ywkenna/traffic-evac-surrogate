# traffic-evac-surrogate/src/collect_results.py

import re
from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# Repo-relative hardcoding
# =========================
EVAC_ONLY_DIR = Path("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/sim-evac-only")
OUT_CSV = Path("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/dataset/collected_metrics.csv")

# fixed constants
N_EVAC_VEHICLES_FIXED = 7000

# expected columns in evac-only trajectory
REQUIRED_COLS = {"vehID", "linkID", "seq", "enterTime", "leaveTime"}

# filename pattern:
# evac-only_sorted_2026shelterseek_0_1_trajectory.csv
FNAME_RE = re.compile(
    r"evac-only_sorted_2026shelterseek_(\d+)_(\d+)_trajectory\.csv$"
)


def _p90(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, 90))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    # fast-ish read with dtype hints
    df = pd.read_csv(
        path,
        usecols=lambda c: c in REQUIRED_COLS,
        dtype={
            "vehID": "int64",
            "linkID": "int64",      # linkID looks numeric in your sample
            "seq": "int64",
            "enterTime": "int64",
            "leaveTime": "int64",
        },
        engine="c",
    )
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{path.name}] missing columns: {missing}")
    return df


def summarize_one_case(csv_path: Path) -> dict:
    """
    Build per-(state_id, policy_id) summary from one evac-only trajectory csv.
    """
    m = FNAME_RE.search(csv_path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {csv_path.name}")

    state_id = int(m.group(1))
    policy_id = int(m.group(2))

    df = _safe_read_csv(csv_path)

    # n_evac_arrived: unique veh IDs present in evac-only file
    n_arrived = int(df["vehID"].nunique(dropna=True))

    # per-vehicle depart/arrive/travel_time
    g = df.groupby("vehID", sort=False, as_index=True)

    depart = g["enterTime"].min()
    arrive = g["leaveTime"].max()
    travel_time = (arrive - depart).astype("int64")

    # path length proxies
    n_rows = g.size().astype("int64")  # number of records (link traversals) per vehicle
    max_seq = g["seq"].max().astype("int64")

    # arrays for stats
    tt = travel_time.to_numpy(dtype="float64", copy=False)
    dep = depart.to_numpy(dtype="float64", copy=False)
    arr = arrive.to_numpy(dtype="float64", copy=False)
    rows = n_rows.to_numpy(dtype="float64", copy=False)
    seqs = max_seq.to_numpy(dtype="float64", copy=False)

    # summary stats
    out = {
        "state_id": state_id,
        "policy_id": policy_id,

        "n_evac_vehicles": N_EVAC_VEHICLES_FIXED,
        "n_evac_arrived": n_arrived,
        "arrival_rate": (n_arrived / N_EVAC_VEHICLES_FIXED) if N_EVAC_VEHICLES_FIXED else np.nan,

        # travel time stats (seconds or sim-time units)
        "tt_mean": float(np.mean(tt)) if tt.size else np.nan,
        "tt_median": float(np.median(tt)) if tt.size else np.nan,
        "tt_p90": _p90(tt),
        "tt_max": float(np.max(tt)) if tt.size else np.nan,
        "tt_min": float(np.min(tt)) if tt.size else np.nan,

        # departure/arrival time stats (absolute sim time)
        "depart_mean": float(np.mean(dep)) if dep.size else np.nan,
        "arrive_mean": float(np.mean(arr)) if arr.size else np.nan,
        "arrive_p90": _p90(arr),

        # route length proxies
        "links_mean": float(np.mean(rows)) if rows.size else np.nan,
        "links_p90": _p90(rows),
        "maxseq_mean": float(np.mean(seqs)) if seqs.size else np.nan,
        "maxseq_p90": _p90(seqs),

        # bookkeeping
        "src_file": csv_path.name,
    }
    return out


def main():
    if not EVAC_ONLY_DIR.exists():
        raise FileNotFoundError(f"evac-only dir not found: {EVAC_ONLY_DIR}")

    csvs = sorted(EVAC_ONLY_DIR.glob("*.csv"))
    targets = [p for p in csvs if FNAME_RE.search(p.name)]

    if not targets:
        raise RuntimeError(
            f"No matching evac-only files in {EVAC_ONLY_DIR}\n"
            f"Expected pattern: evac-only_sorted_2026shelterseek_<state>_<policy>_trajectory.csv"
        )

    rows = []
    for p in targets:
        try:
            rows.append(summarize_one_case(p))
        except Exception as e:
            # keep going but record the failure
            rows.append({
                "state_id": np.nan,
                "policy_id": np.nan,
                "n_evac_vehicles": N_EVAC_VEHICLES_FIXED,
                "n_evac_arrived": np.nan,
                "arrival_rate": np.nan,
                "tt_mean": np.nan,
                "tt_median": np.nan,
                "tt_p90": np.nan,
                "tt_max": np.nan,
                "tt_min": np.nan,
                "depart_mean": np.nan,
                "arrive_mean": np.nan,
                "arrive_p90": np.nan,
                "links_mean": np.nan,
                "links_p90": np.nan,
                "maxseq_mean": np.nan,
                "maxseq_p90": np.nan,
                "src_file": p.name,
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    # If some rows failed to parse state/policy, try to recover from filename
    if df["state_id"].isna().any() or df["policy_id"].isna().any():
        for i, r in df[df["state_id"].isna() | df["policy_id"].isna()].iterrows():
            m = FNAME_RE.search(str(r.get("src_file", "")))
            if m:
                df.loc[i, "state_id"] = int(m.group(1))
                df.loc[i, "policy_id"] = int(m.group(2))

    # enforce numeric
    df["state_id"] = df["state_id"].astype("int64")
    df["policy_id"] = df["policy_id"].astype("int64")

    # ====== 자리 맞춰 나열(중요) ======
    # state_id 0~25, policy_id 0~39 풀 그리드 만든 다음 merge
    full = pd.MultiIndex.from_product(
        [range(0, 26), range(0, 40)],
        names=["state_id", "policy_id"]
    ).to_frame(index=False)

    df = full.merge(df, on=["state_id", "policy_id"], how="left")

    # sort (already in order, but keep explicit)
    df = df.sort_values(["state_id", "policy_id"]).reset_index(drop=True)

    # output path
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote: {OUT_CSV}")
    print(f"[INFO] matched files: {len(targets)}  / grid size: {len(df)}")
    # quick summary
    ok = df["n_evac_arrived"].notna().sum()
    print(f"[INFO] cases with results: {ok} (others are NaN / missing file)")


if __name__ == "__main__":
    main()
