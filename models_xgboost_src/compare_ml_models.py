import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle




# 기존 config에서 설정 가져오기 (가정)
from config_surrogate import DATASET_CSV, Y_COLS, RANDOM_SEED, TEST_SIZE

# X 컬럼 선택 함수 (수정된 로직 반영)
def pick_x_columns(df):
    X_PREFIXES = ["total_assigned", "state_variant_", "policy_", "alloc_", "dist_", "state_zone"]
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in X_PREFIXES):
            cols.append(c)
    return cols

def calculate_wape(y_true, y_pred):
    # 전체 합계 대비 오차의 합 (0이 많은 데이터에 유리)
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100

def main():
    print(f"Loading data from {DATASET_CSV}...")
    df = pd.read_csv(DATASET_CSV)



#시험적으로 추가해봄 >>>

    # 1. 32개 Zone -> 8개 Sector로 합산
    sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for i, sector_name in enumerate(sectors):
        # 각 섹터에 해당하는 zone_id 찾기 (0, 8, 16, 24 등)
        zone_ids = [i + (j * 8) for j in range(4)]
        zone_cols = [f"state_zone{zid}_alloc" for zid in zone_ids]
        
        # 해당 섹터의 총 합계 변수 생성
        df[f"sector_{sector_name}_total"] = df[zone_cols].sum(axis=1)

    # 2. X_PREFIXES 수정
    # 기존 "state_zone" 대신 "sector_"를 사용합니다.
    X_PREFIXES = [
        "total_assigned",
        "sector_",        # 32개 대신 8개만 학습!
        "state_variant_",
        "alloc_",
        "dist_",
    ]


#시험적으로 추가해봄 <<<

    
    x_cols = pick_x_columns(df)
    print(f"Selected {len(x_cols)} X features and {len(Y_COLS)} Y targets.")

    # 데이터 전처리
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    Y = df[Y_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # 1. Random Forest (다중 출력 기본 지원)
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, Y_train)
    rf_pred = rf.predict(X_test)

    # 2. XGBoost (MultiOutputRegressor로 감싸서 사용)
    print("Training XGBoost...")
    xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=RANDOM_SEED)
    xgb = MultiOutputRegressor(xgb_base)
    xgb.fit(X_train, Y_train)
    xgb_pred = xgb.predict(X_test)

    # 결과 비교 출력
    results = []
    
    print("\n" + "="*80)
    print(f"{'Target Variable':<20} | {'Model':<12} | {'R2 Score':<10} | {'WAPE (%)':<10}")
    print("-" * 80)

    for i, col in enumerate(Y_COLS):
        # RF Metrics
        rf_r2 = r2_score(Y_test[:, i], rf_pred[:, i])
        rf_wape = calculate_wape(Y_test[:, i], rf_pred[:, i])
        
        # XGB Metrics
        xgb_r2 = r2_score(Y_test[:, i], xgb_pred[:, i])
        xgb_wape = calculate_wape(Y_test[:, i], xgb_pred[:, i])
        
        print(f"{col:<20} | {'RF':<12} | {rf_r2:>10.4f} | {rf_wape:>10.2f}%")
        print(f"{'':<20} | {'XGB':<12} | {xgb_r2:>10.4f} | {xgb_wape:>10.2f}%")
        print("-" * 80)

    # 전체 평균 성능 요약
    avg_rf_r2 = r2_score(Y_test, rf_pred)
    avg_xgb_r2 = r2_score(Y_test, xgb_pred)
    
    print(f"\n[Summary Average R2 Score]")
    print(f"Random Forest: {avg_rf_r2:.4f}")
    print(f"XGBoost:       {avg_xgb_r2:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()