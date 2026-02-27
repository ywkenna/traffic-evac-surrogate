import pandas as pd

# 1) 헤더 수정: *_to_* → al_*_to_*
columns = [
    "state_id","policy_id",
    "al_0_to_0","al_0_to_1","al_0_to_2","al_0_to_3","al_0_to_4","al_0_to_5",
    "al_6_to_0","al_6_to_1","al_6_to_2","al_6_to_3","al_6_to_4","al_6_to_5",
    "al_7_to_0","al_7_to_1","al_7_to_2","al_7_to_3","al_7_to_4","al_7_to_5",
    "al_8_to_0","al_8_to_1","al_8_to_2","al_8_to_3","al_8_to_4","al_8_to_5",
    "al_9_to_0","al_9_to_1","al_9_to_2","al_9_to_3","al_9_to_4","al_9_to_5",
    "al_14_to_0","al_14_to_1","al_14_to_2","al_14_to_3","al_14_to_4","al_14_to_5",
    "al_15_to_0","al_15_to_1","al_15_to_2","al_15_to_3","al_15_to_4","al_15_to_5",
    "al_16_to_0","al_16_to_1","al_16_to_2","al_16_to_3","al_16_to_4","al_16_to_5",
    "al_17_to_0","al_17_to_1","al_17_to_2","al_17_to_3","al_17_to_4","al_17_to_5",
    "al_21_to_0","al_21_to_1","al_21_to_2","al_21_to_3","al_21_to_4","al_21_to_5",
    "al_22_to_0","al_22_to_1","al_22_to_2","al_22_to_3","al_22_to_4","al_22_to_5",
    "al_23_to_0","al_23_to_1","al_23_to_2","al_23_to_3","al_23_to_4","al_23_to_5",
    "al_24_to_0","al_24_to_1","al_24_to_2","al_24_to_3","al_24_to_4","al_24_to_5",
    "al_25_to_0","al_25_to_1","al_25_to_2","al_25_to_3","al_25_to_4","al_25_to_5",
    "al_29_to_0","al_29_to_1","al_29_to_2","al_29_to_3","al_29_to_4","al_29_to_5",
    "al_30_to_0","al_30_to_1","al_30_to_2","al_30_to_3","al_30_to_4","al_30_to_5",
    "al_31_to_0","al_31_to_1","al_31_to_2","al_31_to_3","al_31_to_4","al_31_to_5",
]

# 2) state_id x policy_id 조합 생성 (25*40=1000행)
rows = [(s, p) for s in range(25) for p in range(40)]
df = pd.DataFrame(rows, columns=["state_id", "policy_id"])

# 3) 나머지 컬럼 추가 (빈 값)
for c in columns[2:]:
    df[c] = pd.NA

# 4) 컬럼 순서 고정
df = df[columns]

# 5) 엑셀 저장
out_path = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/src-sub/state_policy_template_al.xlsx"
df.to_excel(out_path, index=False)

print(f"Saved: {out_path}  (rows={len(df)}, cols={len(df.columns)})")
