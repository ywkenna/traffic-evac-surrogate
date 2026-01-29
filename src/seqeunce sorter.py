import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm # 파일이 많을 경우 진행바 표시 (설치: pip install tqdm)

# 1. 설정
input_pattern = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/sim-outputs/2026shelterseek_12_*/2026shelterseek_*_*_trajectory.csv"
output_prefix = "C:/Users/new/ETRI 김예원/과제data/traffic-evac-surrogate/sim-sorted/sorted_"

# 2. 파일 목록 가져오기 및 확인
input_files = glob.glob(input_pattern)

if not input_files:
    print(f"패턴에 일치하는 파일을 찾을 수 없습니다: {input_pattern}")
else:
    print(f"총 {len(input_files)}개의 파일 처리를 시작합니다.")

    # 3. 루프 실행 (tqdm으로 진행 상황 표시)
    for input_path_str in tqdm(input_files, desc="Processing Files"):
        input_path = Path(input_path_str)
        
        try:
            # 4. CSV 읽기 (메모리 효율을 위해 필요한 경우 low_memory=False 추가)
            df = pd.read_csv(input_path)

            # 5. 데이터 정렬
            # 정렬 기준이 명확하므로 inplace=True를 써서 메모리 점유를 아주 미세하게 줄일 수 있음
            df.sort_values(by=["vehID", "seq"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # 6. 출력 경로 생성 (pathlib 활용)
            output_path = input_path.parent / f"{output_prefix}{input_path.name}"

            # 7. CSV 저장
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"\n error  파일 처리 중 오류 발생: {input_path.name} -> {e}")

    print("\n모든 작업이 완료되었습니다.")