import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# 1. 데이터 로드 및 전처리 (기존 로직 활용)
def load_and_preprocess():
    # 데이터 경로 및 컬럼 설정은 기존 환경에 맞춰 수정하세요.
    df = pd.read_csv("C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/dataset/dataset_final2.csv")
    X_cols = [col for col in df.columns if col.startswith(('alloc_', 'total_assigned', 'dist_', 'state_zone', 'al_'))]
    Y_cols = ["n_evac_arrived", "arrival_rate", "links_mean", "links_p90"]
    
    X = df[X_cols].values.astype(np.float32)
    Y = df[Y_cols].values.astype(np.float32)
    
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def main():
    X_train, X_test, Y_train, Y_test = load_and_preprocess()

    # 결과 저장용 리스트
    results = []

    for i, col_name in enumerate(["n_evac_arrived", "arrival_rate", "links_mean", "links_p90"]):
        y_train_single = Y_train[:, i]
        y_test_single = Y_test[:, i]

        # --- [Baseline 1] Mean Predictor ---
        mean_val = np.mean(y_train_single)
        y_pred_mean = np.full_like(y_test_single, mean_val)
        r2_mean = r2_score(y_test_single, y_pred_mean)
        rmse_mean = np.sqrt(mean_squared_error(y_test_single, y_pred_mean))

        # --- [Baseline 2] Multiple Linear Regression (MLR) ---
        mlr_model = LinearRegression()
        mlr_model.fit(X_train, y_train_single)
        y_pred_mlr = mlr_model.predict(X_test)
        r2_mlr = r2_score(y_test_single, y_pred_mlr)
        rmse_mlr = np.sqrt(mean_squared_error(y_test_single, y_pred_mlr))

        # --- [Proposed] MLP (기존 결과값 입력) ---
        # 실제 환경에서는 모델 로드 후 예측 수행
        mlp_r2_list = [0.35, 0.34, 0.55, 0.46] # 예시 수치
        mlp_rmse_list = [323.5, 0.045, 3.892, 2.421]
        
        results.append({
            "Metric": col_name,
            "Mean_R2": r2_mean,
            "MLR_R2": r2_mlr,
            "MLP_R2": mlp_r2_list[i],
            "MLR_Improvement_vs_Mean": ((rmse_mean - rmse_mlr) / rmse_mean) * 100,
            "MLP_Improvement_vs_MLR": ((rmse_mlr - mlp_rmse_list[i]) / rmse_mlr) * 100
        })

    # 결과 출력
    result_df = pd.DataFrame(results)
    print(result_df)

if __name__ == "__main__":
    main()