import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정 (타임즈 뉴로만 폰트 적용)
plt.rcParams["font.family"] = "Times New Roman"

metrics = ['Arrival Count', 'Arrival Rate', 'Mean Path', 'p90 Path']
mlr_r2 = [0.2106, 0.2106, 0.2397, 0.1558]
mlp_r2 = [0.2570, 0.2586, 0.3396, 0.2722]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, mlr_r2, width, label='Baseline (MLR)', color='#aec7e8', edgecolor='black')
rects2 = ax.bar(x + width/2, mlp_r2, width, label='Proposed (MLP)', color='#1f77b4', edgecolor='black')

# 레이블 및 제목
ax.set_ylabel('R-squared Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: MLR vs. MLP', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.45) # 상단 여유 공간

# 수치 표시
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.show()