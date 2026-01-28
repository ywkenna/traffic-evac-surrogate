import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# 1. 논문용 폰트 및 스타일 설정 (8pt Times New Roman)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.figsize": (4.5, 3.5) # 논문에 적합한 사이즈
})

def plot_policy_tradeoff(csv_path):
    df = pd.read_csv(csv_path)
    
    # 데이터 준비 (시뮬레이션 실제값 기준)
    # 모델 예측값으로 그리려면 'links_mean_pred' 등으로 컬럼명을 바꾸세요.
    x = df['dist_to_assigned_shelter_mean']
    y = df['links_mean']
    c = df['n_evac_arrived']

    fig, ax = plt.subplots()
    
    # 산점도: 색상(c)은 대피 완료 인원을 의미함
    # cmap='viridis' 또는 'RdYlGn' (빨강-노랑-초록) 추천
    scatter = ax.scatter(x, y, c=c, cmap='RdYlGn', s=15, alpha=0.6, edgecolors='none')
    
    # 컬러바 설정 (대피 성공 인원 표시)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Arrived Vehicles', fontweight='bold', rotation=270, labelpad=10)

    # 축 설정
    ax.set_xlabel('Mean Distance to Assigned Shelter (Normalized)', fontweight='bold')
    ax.set_ylabel('Mean Path Length Proxy (Links)', fontweight='bold')
    #ax.set_title('Interaction between Allocation Distance, Path Complexity, and Efficiency')
    
    ax.grid(True, linestyle=':', alpha=0.5)

    # ---------------------------------------------------------
    # [논문 포인트] 비효율 구간 주석 예시 (수동 위치 조정 필요)
    # 거리는 짧은데(왼쪽), 링크는 높은(위쪽) 구간 = 병목 및 정체 구간
    # ---------------------------------------------------------
    #ax.annotate('Inefficient Assignment\n(Congested)', 
    #            xy=(x.min()*1.1, y.max()*0.9), 
    #            xytext=(x.min()*1.5, y.max()*0.95),
    #            arrowprops=dict(facecolor='black', arrowstyle='->'),
    #            fontsize=7, color='red')

    plt.tight_layout()
    plt.savefig("./plots(4y)/policy_interaction_plot.pdf", dpi=600)
    plt.show()

# 실행
plot_policy_tradeoff("./dataset/dataset_final.csv")