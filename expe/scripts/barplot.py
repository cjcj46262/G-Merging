import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys


# 数据

data = np.array([
    [68.9, 76.2, 76.4, 72.4, 77.5, 77.0, 77.2, 78.0],   # Tox21
    [63.3, 63.8, 66.5, 65.0, 65.6, 65.0, 65.7, 64.8],   # Toxcast
    [58.1, 64.6, 63.7, 61.3, 63.2, 64.1, 64.6, 62.5],   # SIDER
    [61.9, 66.9, 77.7, 72.4, 77.5, 76.5, 76.0, 74.0],   # ClinTox
    [55.3, 65.8, 68.5, 67.0, 68.9, 67.6, 67.0, 69.6],   # BBBP
    [78.5, 84.6, 78.1, 78.9, 79.3, 83.5, 86.6, 86.8],   # BACE
    [59.1, 73.9, 71.2, 73.7, 75.1, 73.9, 76.0, 79.6],   # HIV
    [72.3, 82.0, 76.8, 70.7, 78.2, 80.7, 80.8, 83.9],   # MUV
    [64.7, 72.2, 72.4, 70.2, 73.2, 73.5, 74.2, 74.9]    # Average
])
datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV']
dataid = 8
ave = data[dataid]
x_range = [1,2,3,4,5,6,7,8]
x_labels = [
    'Pretrained', 'w/o parameter merging', 'w/o node level moe adapters',
    'w/o graph level moe adapters', 'w/o TWD', 'w/o L1 distance', 'G-Merging (Ours)', 'Full Fine-Tuned'
]
bar_width=0.8

# 为每个柱子指定不同的颜色
colors = ['#D3D3D3', '#FFB3BA', '#FFCC99', '#A7C7E7', '#B5E7A0', '#D3B8AE', 'orange', '#696969']

# plt.figure(figsize=(8, 5.5))
# # 绘制柱状图
# plt.bar(x_range, ave, bar_width, color=colors, edgecolor='black', linewidth=0.8, label=x_labels)
# # plt.ylim(62.5, 76.5)
# y_min = min(ave)
# y_max = max(ave)
# padding = (y_max - y_min) * 0.1  # 上下各留 10% 空间
# plt.ylim(y_min - padding, y_max + padding)
#
# for x, y in zip(x_range, ave):
#     plt.text(x, y + 0.3, f'{y:.1f}', ha='center', va='bottom', fontsize=15)
#
#
# ax = plt.gca()  # 获取当前坐标轴
# ax.tick_params(axis='y', labelsize=17)  # 设置纵坐标刻度字体大小
#
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # 添加标题和标签
# plt.title(f'{datasets[dataid]}', fontsize=20)
# # plt.xlabel('methods', fontsize=15)
# plt.ylabel('Avg. ROC scores', fontsize=20)
# plt.xticks([])
# # plt.legend(loc='upper left', fontsize=17, bbox_to_anchor=(-0.15, 0), ncol=2)


# 构造图例元素（每个颜色一个图块）
legend_handles = [Patch(facecolor=color, edgecolor='black', label=label) for color, label in zip(colors, x_labels)]

# 创建空白图，只画图例
fig, ax = plt.subplots(figsize=(8, 5.5))  # 可以调整高度
ax.axis('off')  # 关闭坐标轴

# 添加图例
ax.legend(
    handles=legend_handles,
    loc='center',
    ncol=1,              # 每行显示几个图例项
    fontsize=20,
    frameon=False        # 不加边框
)

# 显示图形
plt.savefig(f"ablation_bar_legend.pdf", dpi=300, bbox_inches='tight', pad_inches=0)



'''
# 数据
datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV']
Weight_Average = [74.7, 64.5, 60.4, 70.7, 63.5, 78.8, 66.5, 77.5]
AGMM = [77.4, 65.8, 64.8, 74.2, 67.1, 86.8, 74.2, 81.9]

# 设置柱状图位置和宽度
x = np.arange(len(datasets))  # x轴位置
width = 0.35  # 柱宽

# 创建图形
fig, ax = plt.subplots(figsize=(7.2, 6))

# 画柱子
bar1 = ax.bar(x - width/2, Weight_Average, width, label='Weight Average', color='#DAA520')
bar2 = ax.bar(x + width/2, AGMM, width, label='G-Merging(ours)', color='#B22222')
ax.set_ylim(50, 90)

# 添加标签和标题
ax.set_ylabel('ROC scores', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# ax.set_title('Performance of Weight Average and AGMM')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, fontsize=20)
ax.legend(fontsize=24, loc='upper left')

# # 添加数值标签（可选）
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# add_labels(bar1)
# add_labels(bar2)

# 显示图形
plt.tight_layout()
plt.savefig("WeiAve_GMerging_bar.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
'''

'''
# 数据
datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV', 'Average']
agmm_a = [77.4, 65.8, 65.0, 73.9, 67.1, 86.8, 74.0, 81.4, 74.0]
agmm_a2 = [77.3, 65.8, 64.6, 74.5, 66.8, 86.6, 74.0, 80.5, 73.8]
agmm_a3 = [77.0, 65.7, 64.1, 74.6, 66.6, 85.8, 75.1, 80.6, 73.9]

# 柱子位置参数
x = np.arange(len(datasets))
width = 0.25  # 每根柱子的宽度

# 创建画布
fig, ax = plt.subplots(figsize=(14, 6))

# 画柱子（每个位置三根）
bar1 = ax.bar(x - width, agmm_a, width, label=r'G-Merging($A$)', color='#5B4B8A', edgecolor='#4F4F4F')           # 金色
bar2 = ax.bar(x, agmm_a2, width, label=r'G-Merging($\operatorname{sign}(A^2)$)', color='#7D6BB3', edgecolor='#4F4F4F') # 深红色
bar3 = ax.bar(x + width, agmm_a3, width, label=r'G-Merging($\operatorname{sign}(A^3)$)', color='#A698D6', edgecolor='#4F4F4F') # 钢蓝色

# 设置坐标轴
ax.set_ylabel('ROC Scores', fontsize=16)
ax.set_title('Performance on Different Adjacency Matrices', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)

ax.bar_label(bar1, padding=3, fontsize=10, color='black')
ax.bar_label(bar2, padding=3, fontsize=10, color='black')
ax.bar_label(bar3, padding=3, fontsize=10, color='black')


# 设置纵坐标范围（根据你数据定，可调）
ax.set_ylim(60, 90)

# 加图例
ax.legend(fontsize=16)

# 自动调整布局
plt.tight_layout()

# 展示图
plt.savefig("adj_matrix_bar.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
'''