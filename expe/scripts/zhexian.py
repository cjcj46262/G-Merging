import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import re
# import sys
'''
# 设置绘图风格
# sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 16})

list_datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV']

# topk对应的横坐标
# hyper_lambda = np.arange(0.05, 0.275, 0.025)
hyper_alpha = np.arange(0,8,1)
xticklabels = ['1.e-06', '1.e-05', '1.e-04', '1.e-03', '1.e-02', '1.e-01', '1.e+00', '1.e+01']
# print(hyper_alpha)
# sys.exit()


list_file_name = [
    # 'AGMM.txt',
    # 'AGMM_1e+00_0.05.txt',
    # 'AGMM_1e+00_0.075.txt',
    # 'AGMM_1e+00_0.1.txt',
    # 'AGMM_1e+00_0.125.txt',
    # 'AGMM_1e+00_0.15.txt',
    # 'AGMM_1e+00_0.175.txt',
    # 'AGMM_1e+00_0.2.txt',
    # 'AGMM_1e+00_0.225.txt',
    # 'AGMM_1e+00_0.25.txt',

    'AGMM_1e-06_1.txt',
    'AGMM_1e-05_1.txt',
    'AGMM_1e-04_1.txt',
    'AGMM_1e-03_1.txt',
    'AGMM_1e-02_1.txt',
    'AGMM_1e-01_1.txt',
    'AGMM_1e+00_1.txt',
    'AGMM_1e+01_1.txt',

]

all_scores = []
for file_name in list_file_name:
    scores = []
    with open(f'./shell1/gcn_supervised_contextpred/{file_name}', 'r') as file:
        for line in file:
            match = re.search(r'Test ROC AUC score: ([\d\.]+)', line)
            if match:
                scores.append(float(match.group(1)))
    all_scores.append(scores)

score_np = np.array(all_scores)
# print(score_np)
# score_np = score_np.T



# 绘图
plt.figure(figsize=(8, 10))
for i, dataset in enumerate(list_datasets):
    plt.plot(hyper_alpha, score_np[:, i], marker='h', label=dataset, markersize=10)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

plt.xlabel('Hyper Parameters', fontsize=16)
plt.ylabel('ROC Score',fontsize=16)
# plt.title('Performance across Different Top-k in MoE')
plt.xticks(hyper_alpha, xticklabels)

plt.grid(True)
plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(0.02, 0.49), ncol=2)

plt.tight_layout()
plt.savefig(f"hyper_zhexian_alpha_gcncontext.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
'''

'''
new_list_datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV', 'Average']
# Data for the average values and the corresponding methods' times
rank = ['10', '20', '30', '40', '50']
# rank = ['1', '2', '3', '4', '5', '6', '7', '8']
data = [72.9, 73.0, 73.9, 73.9, 73.8]
upper = [74.9,74.9,74.9,74.9,74.9]
lower = [64.7,64.7,64.7,64.7,64.7]


# data_by_column = {
#     "col_1": [78.0, 68.9, 77.2, 77.6, 77.5, 77.8, 77.9, 77.9, 77.9, 77.9],
#     "col_2": [64.8, 63.3, 65.7, 65.9, 65.8, 65.4, 65.5, 65.8, 65.9, 65.9],
#     "col_3": [62.5, 58.1, 64.6, 64.7, 62.7, 61.9, 61.1, 61.9, 62.9, 63.9],
#     "col_4": [74.0, 61.9, 73.3, 74.6, 74.2, 73.9, 75.4, 75.2, 75.2, 75.3],
#     "col_5": [69.6, 55.3, 67.0, 66.7, 66.8, 67.0, 66.8, 66.9, 66.9, 66.9],
#     "col_6": [86.8, 78.5, 87.0, 85.4, 85.7, 84.8, 84.0, 86.1, 86.2, 86.2],
#     "col_7": [79.6, 59.1, 74.1, 73.7, 73.3, 72.7, 72.9, 73.3, 73.9, 73.9],
#     "col_8": [83.9, 72.3, 81.1, 80.5, 80.6, 80.5, 80.6, 80.7, 80.7, 80.7],
#     "col_9": [74.9, 64.7, 73.8, 73.6, 73.3, 73.0, 73.0, 73.5, 73.7, 73.8]
# }

# dataset_id = 6  
# data = data_by_column[f'col_{dataset_id}'][2:]
# upper = [data_by_column[f'col_{dataset_id}'][0]] * 8
# lower = [data_by_column[f'col_{dataset_id}'][1]] * 8


# Plotting the average values as a line plot
plt.figure(figsize=(4.4, 3.2))
plt.plot(rank, data, marker='D', color='orange', linestyle='-', linewidth=2.3, markersize=9, )
plt.plot(rank, upper, marker='x', color='black', linestyle='--', linewidth=2.3, markersize=9, )
plt.plot(rank, lower, marker='x', color='grey', linestyle='--', linewidth=2.3, markersize=9, )
plt.text(rank[0], upper[0]+0.3, "Upper Bound (Full Fine-Tuned)", fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='left')
plt.text(rank[0], lower[0]+0.3, "Lower Bound (Pretrained)", fontsize=12, color='grey', verticalalignment='bottom', horizontalalignment='left')
plt.text(rank[0], data[0]-0.3, "G-Merging", fontsize=12, color='orange', verticalalignment='top', horizontalalignment='left')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)


# Adding labels and title
plt.xlabel('rank', fontsize=12)
plt.ylabel('Avg. ROC scores', fontsize=12)
# plt.title(new_list_datasets[dataset_id-1], fontsize=14,pad=20)
# plt.xticks(rotation=45)
plt.grid(True)

# upper_bound = 74.9
# lower_bound = 64.7
# plt.axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper Bound (Full Fine-Tuned) = {upper_bound}')
# plt.axhline(y=lower_bound, color='g', linestyle='--', label=f'Lower Bound (Pretrained) = {lower_bound}')
# # Displaying the plot
plt.tight_layout()
plt.savefig("r_zhexian_rank.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
'''




'''
new_list_datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV', 'Average']
# Data for the average values and the corresponding methods' times
# rank = ['10', '20', '30', '40', '50']
rank = ['1', '2', '3', '4', '5', '6', '7', '8']
# average_values = [72.9, 73.0, 73.9, 73.9, 73.8]
# upper = [74.9,74.9,74.9,74.9,74.9]
# lower = [64.7,64.7,64.7,64.7,64.7]


data_by_column = {
    "col_1": [78.0, 68.9, 77.2, 77.6, 77.5, 77.8, 77.9, 77.9, 77.9, 77.9],
    "col_2": [64.8, 63.3, 65.7, 65.9, 65.8, 65.4, 65.5, 65.8, 65.9, 65.9],
    "col_3": [62.5, 58.1, 64.6, 64.7, 62.7, 61.9, 61.1, 61.9, 62.9, 63.9],
    "col_4": [74.0, 61.9, 73.3, 74.6, 74.2, 73.9, 75.4, 75.2, 75.2, 75.3],
    "col_5": [69.6, 55.3, 67.0, 66.7, 66.8, 67.0, 66.8, 66.9, 66.9, 66.9],
    "col_6": [86.8, 78.5, 87.0, 85.4, 85.7, 84.8, 84.0, 86.1, 86.2, 86.2],
    "col_7": [79.6, 59.1, 74.1, 73.7, 73.3, 72.7, 72.9, 73.3, 73.9, 73.9],
    "col_8": [83.9, 72.3, 81.1, 80.5, 80.6, 80.5, 80.6, 80.7, 80.7, 80.7],
    "col_9": [74.9, 64.7, 73.8, 73.6, 73.3, 73.0, 73.0, 73.5, 73.7, 73.8]
}

dataset_id = 9
data = data_by_column[f'col_{dataset_id}'][2:]
upper = [data_by_column[f'col_{dataset_id}'][0]] * 8
lower = [data_by_column[f'col_{dataset_id}'][1]] * 8


# Plotting the average values as a line plot
plt.figure(figsize=(5, 3.8))
plt.plot(rank, data, marker='D', color='#6495ED', linestyle='-', linewidth=2.3, markersize=10, label = 'AGMM(rank)')
plt.plot(rank, upper, marker='x', color='black', linestyle='--', linewidth=2.3, markersize=10, )
plt.plot(rank, lower, marker='x', color='grey', linestyle='--', linewidth=2.3, markersize=10, )
plt.text(rank[0], upper[0]+0.3, "Upper Bound (Full Fine-Tuned)", fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='left')
plt.text(rank[0], lower[0]+0.3, "Lower Bound (Pretrained)", fontsize=12, color='grey', verticalalignment='bottom', horizontalalignment='left')
plt.text(rank[0], data[0]+0.8, "G-Merging", fontsize=12, color='#6495ED', verticalalignment='top', horizontalalignment='left')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)


# Adding labels and title
plt.xlabel('topk', fontsize=14)
plt.ylabel('Avg. ROC scores', fontsize=14)
# plt.title(new_list_datasets[dataset_id-1], fontsize=14,pad=20)
# plt.xticks(rotation=45)
plt.grid(True)

# upper_bound = 74.9
# lower_bound = 64.7
# plt.axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper Bound (Full Fine-Tuned) = {upper_bound}')
# plt.axhline(y=lower_bound, color='g', linestyle='--', label=f'Lower Bound (Pretrained) = {lower_bound}')
# # Displaying the plot
plt.tight_layout()
plt.savefig(f"r_zhexian_together.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
'''


# 数据部分
rank1 = ['10', '20', '30', '40', '50']
data1 = [72.9, 73.0, 73.9, 73.9, 73.8]
upper1 = [74.9]*5
lower1 = [64.7]*5

rank2 = ['1', '2', '3', '4', '5', '6', '7', '8']
data_by_column = {
    "col_9": [74.9, 64.7, 73.8, 73.6, 73.3, 73.0, 73.0, 73.5, 73.7, 73.8]
}
data2 = data_by_column["col_9"][2:]
upper2 = [data_by_column["col_9"][0]] * 8
lower2 = [data_by_column["col_9"][1]] * 8

# 创建子图（1行2列）
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'wspace': 0.2})  # 总宽度适当增大以容纳两个图


# ----------- 左图 ------------
axes[0].plot(rank1, data1, marker='D', color='orange', linestyle='-', linewidth=2.3, markersize=9)
axes[0].plot(rank1, upper1, marker='x', color='black', linestyle='--', linewidth=2.3, markersize=9)
axes[0].plot(rank1, lower1, marker='x', color='grey', linestyle='--', linewidth=2.3, markersize=9)
axes[0].text(rank1[0], upper1[0]+0.3, "Upper Bound (Full Fine-Tuned)", fontsize=16, color='black', verticalalignment='bottom')
axes[0].text(rank1[0], lower1[0]+0.3, "Lower Bound (Pretrained)", fontsize=16, color='grey', verticalalignment='bottom')
axes[0].text(rank1[0], data1[0]-0.45, "G-Merging", fontsize=16, color='orange', verticalalignment='top')
axes[0].set_xlabel('rank', fontsize=19)
axes[0].set_ylabel('Avg. ROC scores', fontsize=16)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(True)
axes[0].tick_params(axis='both', labelsize=16)

# ----------- 右图 ------------
axes[1].plot(rank2, data2, marker='D', color='#6495ED', linestyle='-', linewidth=2.3, markersize=10)
axes[1].plot(rank2, upper2, marker='x', color='black', linestyle='--', linewidth=2.3, markersize=10)
axes[1].plot(rank2, lower2, marker='x', color='grey', linestyle='--', linewidth=2.3, markersize=10)
axes[1].text(rank2[0], upper2[0]+0.3, "Upper Bound (Full Fine-Tuned)", fontsize=16, color='black', verticalalignment='bottom')
axes[1].text(rank2[0], lower2[0]+0.3, "Lower Bound (Pretrained)", fontsize=16, color='grey', verticalalignment='bottom')
axes[1].text(rank2[2], data2[2]-0.8, "G-Merging", fontsize=16, color='#6495ED', verticalalignment='top')
axes[1].set_xlabel('top-k', fontsize=19)
axes[1].set_ylabel('Avg. ROC scores', fontsize=16)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(True)
axes[1].tick_params(axis='both', labelsize=16)

# 自动调整布局，防止重叠
plt.tight_layout()

# 保存为PDF
plt.savefig("r_zhexian_combined.pdf", dpi=300, bbox_inches='tight', pad_inches=0)






