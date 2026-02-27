# [원자력발전소 주변 PAZ 구역 차량 대피소 할당 정책 비교를 위한 Surrogate Model의 설계]

## 연구 배경 및 목적
* **배경:** 기존 UNIQ-SALT-TGE 시뮬레이션은 개별 Agent의 경로를 구현하는 시뮬레이터로써 높은 성능을 가져 Agent의 미시적 행동 구현에 있어 탁월한 성능을 가지나, 수십~수백만개의 다양한 정책을 비교하고자 할 경우 높은 computational cost가 부담이 됨.
* **목적:** 대피소 할당 정책의 결과는 {초기 차량 수요(차량 수) 및 위치, 대피 장소 및 할당 방식} 등 여러 변수에 의해 다양하게 변화함. 빠른 속도의 정책 비교를 달성하기 위해, UNIQ-SALT-TGE의 주요 대피 성능 효율 지표만을 학습 및 예측하는 Surrogate Model을 구현하고자 하였으며, 이를 다양한 대피 정책 간 상대적 성능 비교를 빠르게 수행하기 위한 1차적 필터링ㆍ스크리닝 도구로 제시함.

## 주요 내용
* 초기 상태 25종, 대피소 6곳, 대피 정책 40개를 합쳐 구성된 1000개의 시나리오의 시뮬레이션 데이터를 인풋으로 받아 학습 데이터 생성.
* 생성된 학습 데이터는 다층 퍼셉트론 구조로 학습됨.
* 상태, 정책 및 시뮬레이션 시나리오 생성을 위한 코드는 포함되어있지 않음

## 실행 방법

* A. 준비된 학습 데이터셋만 모델 학습하는 경우(학습 데이터 생성 과정 생략)
* 1. dataset/dataset_final2.csv 확인(데이터셋이므로 필요 시 변경).
  2. model_src/config_surrogate.py 의 DATASET_CSV, MODEL_DIR 경로 수정.
  3. model_src/train_surrogate_ver2.py 실행.

* B. 학습 데이터셋 구축 후 실행하는 경우
* 1. sim-outputs에 시뮬레이션 결과 저장(폴더로).
* 2. src/ 내에 위치한 collect_results.py -> build_dataset.py -> data_add_dist.py -> dataset_final_generator.py 순으로 실행(4개 파일 모두 환경에 맞게 경로 수정 필요).
* 3. 2과정 이후 생성된 dataset_final.csv 에는 allocation 정보(구역-대피소 쌍의 대피 인원)가 존재하지 않으므로 수동으로 추가해주어야 함.
  4. src-al-generator/policy-state-allocationt-template-generator.py 실행하여 state_policy_template.xlsx 생성.
  5. src-al-generator/policy-state-allocation.py 실행하여 policy-state-allocation.xlsx 생성 후, 해당 파일 내용 복사하여 dataset_final 뒤에 추가.
  6. A. 과정 통하여 모델 학습 및 확인.
 
* 그 외.
  - baseline_MLR.py를 사용하여 MLR과의 성능 비교 가능(모델 성능 입력 필요)
  - plotting 관련 코드는 result-plotter 에 위치

## 📦 파일 구조
┣ 📂 baselineMLR
┣ ┗ 📜 baseline_MLR.py
┣ 📂 data
┃ ┣ 📜 **case1000_zone_shelter_alloc_cap2000.csv**
┃ ┣ 📜 evac-area.xlsx
┃ ┣ 📜 policy40_shelter_ratio.csv
┃ ┣ 📜 **shelter_zone_dist.csv**
┃ ┗ 📜 state25_zone32_N7000_pmz0_5km.csv
┣ 📂 dataset
┃ ┣ 📂 processed
┃ ┃ ┗ 📜 dataset.csv
┃ ┣ 📜 collected_metrics.csv
┃ ┣ 📜 dataset_final.csv
┃ ┣ 📜 **dataset_final2.csv**
┃ ┗ 📜 dataset_with_dist.csv
┣ 📂 model_src
┃ ┣ 📂 __pycache__
┃ ┃ ┣ 📜 config_surrogate.cpython-314.pyc
┃ ┃ ┗ 📜 train_surrogate.cpython-314.pyc
┃ ┣ 📜 config_surrogate.py
┃ ┣ 📜 predict_surrogate.py
┃ ┣ 📜** train_surrogate_ver2.py**
┃ ┣ 📜 train_surrogate.py
┃ ┗ 📜 train_surrogate2.py
┣ 📂 models
┣ 📂 models_xgboost
┣ 📂 models_xgboost_src
┣ 📂 result-plotter
┣ 📂 sim-evac-only
┣ 📂 sim-outputs
┣ 📂 sim-sorted
**┣ 📂 src**
┃ ┣ 📜 build_dataset.py
┃ ┣ 📜 collect_results.py
┃ ┣ 📜 dataset_add_dist.py
┃ ┣ 📜 dataset_final_generator.py
┃ ┣ 📜 evac_only_filter.py
┃ ┗ 📜 seqeunce sorter.py
┣ 📂 src-al-generator
┃ ┣ 📜 filled_state_policy_template_al.xlsx
┃ ┣ 📜 policy-state-allocation-template-gene...
┃ ┣ 📜 policy-state-allocation.py
┗ ┗ 📜 state_policy_template_al.xlsx
