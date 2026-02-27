import pandas as pd

TEMPLATE_XLSX = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/src-sub/state_policy_template_al.xlsx"
ALLOC_CSV     = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/data/case1000_zone_shelter_alloc_cap2000.csv"
OUT_XLSX      = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/src-sub/filled_state_policy_template_al.xlsx"

# 1) 템플릿 로드
tmpl = pd.read_excel(TEMPLATE_XLSX)

# 2) allocation csv 로드
alloc = pd.read_csv(ALLOC_CSV)

# (필수 컬럼 체크)
required_cols = {"state_id", "policy_id", "zone32_id", "shelter_id", "assigned_vehicles"}
missing = required_cols - set(alloc.columns)
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

# 3) CSV에서 템플릿 컬럼명 형태로 만들기: al_{zone32_id}_to_{shelter_id}
alloc = alloc.copy()
alloc["col"] = "al_" + alloc["zone32_id"].astype(int).astype(str) + "_to_" + alloc["shelter_id"].astype(int).astype(str)

# 4) 혹시 같은 (state,policy,zone,shelter)가 여러 줄이면 합쳐서 1개로
alloc_agg = (
    alloc.groupby(["state_id", "policy_id", "col"], as_index=False)["assigned_vehicles"]
    .sum()
)

# 5) wide 형태로 피벗: index=(state_id,policy_id), columns=al_*_to_*, values=assigned_vehicles
wide = alloc_agg.pivot(index=["state_id", "policy_id"], columns="col", values="assigned_vehicles")

# 6) 템플릿에 업데이트 (템플릿에 존재하는 al_*_to_* 컬럼만 채움)
result = tmpl.set_index(["state_id", "policy_id"])
common_cols = [c for c in wide.columns if c in result.columns]

# 채워넣기: wide에 값이 있는 곳만 덮어쓰기
result.loc[wide.index, common_cols] = wide[common_cols]

# (선택) NaN을 0으로 채우고 싶으면 아래 주석 해제
al_cols = [c for c in result.columns if c.startswith("al_") and "_to_" in c]
result[al_cols] = result[al_cols].fillna(0)

# 7) 저장
result = result.reset_index()
result.to_excel(OUT_XLSX, index=False)
print(f"Saved: {OUT_XLSX} (rows={len(result)}, cols={len(result.columns)})")
