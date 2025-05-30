import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

weights_np = np.array([[78.0, 62.1, 58.5, 64.0, 57.7, 51.0, 58.2, 68.0],
              [73.0, 64.8, 59.7, 53.3, 63.4, 70.8, 54.7, 66.5],
              [68.9, 63.9, 62.5, 55.2, 54.9, 78.0, 59.4, 70.3],
              [60.2, 57.4, 50.2, 74.0, 55.0, 57.6, 51.4, 70.3],
              [56.0, 54.6, 54.7, 55.2, 69.6, 40.6, 33.6, 62.4],
              [62.1, 59.7, 53.9, 62.3, 53.0, 86.8, 63.1, 72.0],
              [56.6, 54.2, 53.7, 48.8, 46.9, 59.2, 79.6, 69.4],
              [52.9, 56.0, 51.0, 50.5, 47.7, 62.7, 62.6, 83.9]]).T


# 3. 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(weights_np, cmap="YlGnBu", annot=False, fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 1.0})

new_list_datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV']
y_labels = new_list_datasets
x_labels = new_list_datasets

plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=45, fontsize=20)
plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels, rotation=0, fontsize=20)

cbar = plt.gcf().axes[-1]  # 通常 colorbar 是最后一个 axis
cbar.tick_params(labelsize=18)  # 设置 colorbar 刻度字体大小为 16
# cbar.set_ylabel('ROC Score', fontsize=18)  # 可选，设置 colorbar 的标签字体大小


# 4. 添加标题
plt.title("ROC scores", fontsize=20)
plt.xlabel("models:                                                                     ", fontsize=20, labelpad=-50)
# plt.ylabel("Name of the testing task", fontsize=15)

# 5. 保存图片
plt.savefig(f"heatmap_ft.pdf", dpi=300, bbox_inches='tight',
            pad_inches=0)  # 保存为 PNG 格式，300 DPI

# 6. 关闭图像以释放内存
plt.close()

