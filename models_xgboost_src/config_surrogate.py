from pathlib import Path

# ===== 경로(하드코딩) =====
DATASET_CSV = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\dataset\dataset_final.csv")
MODEL_DIR   = Path(r"C:\Users\new\ETRI 김예원\과제data\traffic-evac-surrogate\models_xgboost")

# ===== 컬럼 정의 =====
DROP_COLS = [
    "src_file",  # 누수 방지
]

# 입력(X)에서 반드시 포함시키고 싶은 컬럼 prefix들
X_PREFIXES = [
    "state_variant_",
    "policy_",
    "alloc_",
    "total_assigned",
    "dist_",
    "state_zone",   # state_zone0_alloc ~ state_zone31_alloc
]

# 출력(Y) 컬럼들
Y_COLS = [
    "n_evac_arrived",
    "arrival_rate",
    "tt_mean", "tt_median", "tt_p90", "tt_max", "tt_min",
    "depart_mean",
    "arrive_mean", "arrive_p90",
    "links_mean", "links_p90",
    "maxseq_mean", "maxseq_p90",
]

RANDOM_SEED = 42
TEST_SIZE = 0.2

# 학습 하이퍼파라미터
EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIMS = [64, 128, 64]
DROPOUT = 0.4
