# G-Merging 架构与数据流讲解文档

---

## 一、待融合模型（微调模型）的架构

G-Merging 的待融合模型是 `GNN_graphpred`，由 GNN 主干 + 图池化 + 分类头组成。
每个任务对应一个独立微调的 `GNN_graphpred` 实例，共 8 个任务（tox21/toxcast/sider/clintox/bbbp/bace/hiv/muv）。

### 1.1 整体结构

```
输入分子图
    │
    ▼
[GNN 主干] × 5 层（GINConv + BatchNorm + Dropout）
    │  节点嵌入：[N, 300]
    ▼
[图池化层] global_mean_pool
    │  图嵌入：[bs, 300]
    ▼
[分类头] Linear(300 → num_tasks)
    │  logits：[bs, num_tasks]
    ▼
输出预测
```

### 1.2 输入嵌入层（GNN.__init__）

| 层名 | 类型 | 输入维度 | 输出维度 | 作用 |
|------|------|----------|----------|------|
| `x_embedding1` | Embedding(120, 300) | 原子类型索引 [N] | [N, 300] | 将原子序数（1~119）映射为 300 维嵌入 |
| `x_embedding2` | Embedding(3, 300) | 手性标签索引 [N] | [N, 300] | 将手性标签（3类）映射为 300 维嵌入 |

两者相加得到初始节点特征：`x = x_embedding1(x[:,0]) + x_embedding2(x[:,1])`，输出 `[N, 300]`。

### 1.3 GNN 消息传递层（5 层 GINConv）

每层结构（以 GINConv 为例）：

| 子层 | 类型 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| `edge_embedding1` | Embedding(6, 300) | 键类型索引 | [E, 300] | 将键类型（单/双/三/芳香等）映射为边嵌入 |
| `edge_embedding2` | Embedding(3, 300) | 键方向索引 | [E, 300] | 将键方向映射为边嵌入 |
| 消息聚合 | MessagePassing | [N, 300] + [E, 300] | [N, 300] | 邻居节点特征 + 边特征求和后聚合 |
| `mlp` | Linear(300→600)→ReLU→Linear(600→300) | [N, 300] | [N, 300] | 非线性变换，更新节点表示 |
| `batch_norms[i]` | BatchNorm1d(300) | [N, 300] | [N, 300] | 批归一化，稳定训练 |
| Dropout | Dropout(0.5) | [N, 300] | [N, 300] | 正则化（最后一层不加 ReLU） |

5 层 GINConv 串联，每层输出 `[N, 300]`，最终取最后一层输出（JK="last"）作为节点表示。

### 1.4 图池化层

| 层名 | 类型 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| `pool` | global_mean_pool | [N, 300] + batch向量 | [bs, 300] | 对每个图的所有节点特征取均值，得到图级表示 |

### 1.5 分类头

| 层名 | 类型 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| `graph_pred_linear` | Linear(300, num_tasks) | [bs, 300] | [bs, num_tasks] | 线性分类，输出各任务的 logit |

各任务的 `num_tasks`：tox21=12, toxcast=617, sider=27, clintox=2, bbbp=1, bace=1, hiv=1, muv=17。

### 1.6 参数量汇总（以 GIN+contextpred 为例）

| 组件 | 参数量（约） |
|------|-------------|
| 输入嵌入（x_embedding1/2） | 120×300 + 3×300 = 36,900 |
| 5层 GINConv（含边嵌入+MLP） | 5 × (6×300 + 3×300 + 300×600 + 600×300) ≈ 1.35M |
| 5层 BatchNorm | 5 × 300×2 = 3,000 |
| 分类头 | 300 × num_tasks（任务相关） |
| **主干合计** | **约 1.4M** |

---

## 二、融合后模型的完整架构

融合后模型同样是 `GNN_graphpred`，但在 Phase 3 推理时以 `moe=True` 模式初始化，在原有架构基础上新增了 G-Merging 的核心组件。

### 2.1 融合后模型整体结构

```
输入分子图
    │
    ▼
[输入嵌入] x_embedding1 + x_embedding2
    │  [N, 300]
    ▼
┌─────────────────────────────────────────────────────┐
│  GNN 主干（5层，权重来自 Phase 1 Task Arithmetic）    │
│                                                     │
│  for layer in range(5):                             │
│    h = GINConv[layer](h, edge_index, edge_attr)     │
│    │  [N, 300]                                      │
│    ▼                                                │
│  ┌─────────────────────────────────────────────┐   │
│  │  【G-Merging 新增】节点级 MoE Router         │   │
│  │  8个 Expert（NodeAdapter）各自输出 [N, 300]  │   │
│  │  TWD 计算路由分数 → Top-K Softmax 加权融合   │   │
│  │  h = h - weighted_sum(Expert_i(h))          │   │
│  └─────────────────────────────────────────────┘   │
│    │  [N, 300]                                      │
│    ▼                                                │
│    BatchNorm + Dropout                              │
└─────────────────────────────────────────────────────┘
    │  节点表示 [N, 300]
    ▼
[图池化] global_mean_pool → [bs, 300]
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  【G-Merging 新增】图级 MoE Router                   │
│  8个 Expert（GraphAdapter）各自输出 [bs, 300]        │
│  余弦相似度计算路由分数 → Top-K Softmax 加权融合      │
│  graph_rep = graph_rep - weighted_sum(Expert_i(g))  │
└─────────────────────────────────────────────────────┘
    │  [bs, 300]
    ▼
[分类头] Linear(300 → num_tasks) → [bs, num_tasks]
```

### 2.2 G-Merging 新增组件详解

#### 2.2.1 节点级 MoE Expert（surgery_moe_layers）

- **数量**：5层 × 8个 Expert = 40 个 MLP
- **每个 Expert 结构**：`Linear(300→30) → ReLU → Linear(30→300)`（rank=30）
- **参数量**：每个 Expert = 300×30 + 30×300 = 18,000；共 40×18,000 = 720,000

| 维度 | 说明 |
|------|------|
| 输入 | `h`：[N, 300]，当前层 GNN 输出的节点特征 |
| Expert 输出 | `Expert_i(h)`：[N, 300]，第 i 个 Expert 的残差输出 |
| 密集格式 | `moe_dense_list[i]`：[bs, max_N, 300]，用于 TWD 计算 |
| 路由分数 | `moe_score`：[bs, 8]，各 Expert 的 TWD 负距离 |
| 节点路由分数 | `moe_score[batch, :]`：[N, 8]，扩展到节点级 |
| 加权输出 | `h_moe`：[N, 300]，加权融合后的残差 |
| 最终输出 | `h = h - h_moe`：[N, 300] |

#### 2.2.2 图级 MoE Expert（surgery_moe）

- **数量**：8 个 Expert（对应 8 个任务）
- **每个 Expert 结构**：`Linear(300→30) → ReLU → Linear(30→300)`
- **参数量**：8×18,000 = 144,000

| 维度 | 说明 |
|------|------|
| 输入 | `graph_rep`：[bs, 300]，池化后的图表示 |
| Expert 输出 | `Expert_i(graph_rep)`：[bs, 300] |
| 路由分数 | `moe_score`：[bs, 8]，余弦相似度分数 |
| 加权输出 | `graph_rep_moe`：[bs, 300] |
| 最终输出 | `graph_rep = graph_rep - graph_rep_moe`：[bs, 300] |

#### 2.2.3 TWD 路由器（self.TWD）

- **类型**：`GTOTRegularization(order=1)`
- **作用**：在节点级 MoE 中，计算参考 Expert（index=0）与各 Expert 输出之间的拓扑感知 Wasserstein 距离，作为路由分数
- **输入**：两组密集节点特征 [bs, max_N, 300] + 邻接矩阵 [bs, max_N, max_N]
- **输出**：每个图的 Wasserstein 距离向量 [bs]

### 2.3 融合后模型参数量汇总

| 组件 | 参数量 | 来源 |
|------|--------|------|
| GNN 主干（5层 GINConv + 嵌入） | ~1.4M | Phase 1 Task Arithmetic 合并 |
| 分类头 | 300×num_tasks | 从微调模型加载 |
| 节点级 MoE Expert（5层×8） | 720,000 | Phase 2 各任务独立训练 |
| 图级 MoE Expert（8个） | 144,000 | Phase 2 各任务独立训练 |
| **新增 Adapter 合计** | **864,000（约 0.86M）** | 占主干参数约 62% |

---

## 三、以 BBBP 数据集为例的完整数据流

BBBP（Blood-Brain Barrier Penetration）是一个二分类任务，预测分子能否穿透血脑屏障。
- 数据量：约 2,039 个分子
- 标签：`num_tasks=1`，标签值 {1（可穿透）, -1（不可穿透）}
- 划分：scaffold split，训练/验证/测试 ≈ 1,631 / 204 / 204

### 3.1 原始数据 → 图数据（loader.py）

**输入**：SMILES 字符串，例如 `"[Cl].CC(C)NCC(O)c1ccc(Cl)cc1"`

**处理过程**：
1. RDKit 解析 SMILES → 分子对象
2. 遍历原子，提取 2 维特征：`[原子序数索引, 手性标签索引]`
3. 遍历化学键，提取 2 维特征：`[键类型索引, 键方向索引]`（双向，每条键存两次）

**输出**：`torch_geometric.data.Data` 对象

```
data.x          : [N, 2]   long   原子特征矩阵，N=分子原子数（BBBP平均约26个原子）
                            [:,0] = 原子序数索引（0~118，对应H~Og）
                            [:,1] = 手性标签索引（0~2）
data.edge_index : [2, 2E]  long   COO格式边索引，E=化学键数（双向存储，故×2）
                            [0,:] = 源节点索引
                            [1,:] = 目标节点索引
data.edge_attr  : [2E, 2]  long   边特征矩阵
                            [:,0] = 键类型索引（0=单键,1=双键,2=三键,3=芳香键）
                            [:,1] = 键方向索引（0=无,1=向上,2=向下）
data.y          : [1]      float  标签（1 或 -1）
data.id         : [1]      long   分子在数据集中的索引
```

**具体示例**（假设某 BBBP 分子有 26 个原子、28 条化学键）：
```
data.x          : shape [26, 2]
data.edge_index : shape [2, 56]   （28条键 × 2方向）
data.edge_attr  : shape [56, 2]
data.y          : shape [1]       值为 1.0（可穿透血脑屏障）
```

### 3.2 DataLoader 批处理

`DataLoader(train_dataset, batch_size=512, shuffle=True)`

PyG 将一个 batch 内的多个图合并为一个大图（节点/边拼接，batch 向量记录归属）：

```
batch.x          : [N_total, 2]    N_total = batch内所有分子原子数之和（≈512×26=13,312）
batch.edge_index : [2, E_total]    E_total = 所有边数之和（≈512×56=28,672）
batch.edge_attr  : [E_total, 2]
batch.y          : [512, 1]        每个图的标签
batch.batch      : [N_total]       long，每个节点所属图的索引（0~511）
batch.id         : [512]           每个图的数据集索引
```

### 3.3 Phase 1：Task Arithmetic 数据流

**输入**：预训练模型权重 + 8个任务的微调模型权重

```python
# 对每个任务计算 task vector
task_vector_i = finetuned_state_dict_i - pretrained_state_dict
# 形状与对应参数相同，例如：
# gnns.0.mlp.0.weight: [600, 300]  → task_vector[key]: [600, 300]

# 求和
task_vector_sum = Σ task_vector_i   # 各参数形状不变

# 叠加到预训练模型
merged_params[key] = pretrained[key] + 0.2 * task_vector_sum[key]
# 0.2 是 BBBP 对应的 scaling_coef_（gin+contextpred）
```

**输出**：粗粒度合并模型权重字典，形状与预训练模型完全相同。

### 3.4 Phase 2：BBBP Adapter 训练数据流

**模型状态**：
- `model`：加载了 Phase 1 合并权重 + 随机初始化的 NodeAdapter/GraphAdapter（surgery=True）
- `finetune_model`：加载了 BBBP 微调权重（固定，不参与梯度）

**一个训练 batch 的完整数据流**（batch_size=512，N_total≈13,312）：

```
Step 1: 提取微调模型中间层特征（source，detach）
  输入: batch.x [13312, 2], batch.edge_index [2, 28672], batch.edge_attr [28672, 2]

  finetune_model 前向传播：
    x_emb = embedding1(x[:,0]) + embedding2(x[:,1])  → [13312, 300]

    Layer 0 GINConv → mlp.2 输出: [13312, 300]  ← 被 IntermediateLayerGetter 捕获
    Layer 1 GINConv → mlp.2 输出: [13312, 300]  ← 捕获
    Layer 2 GINConv → mlp.2 输出: [13312, 300]  ← 捕获
    Layer 3 GINConv → mlp.2 输出: [13312, 300]  ← 捕获
    Layer 4 GINConv → mlp.2 输出: [13312, 300]  ← 捕获

    graph_rep = mean_pool(h_last, batch)          → [512, 300]

  intermediate_output_s = {
    'gnn.gnns.0.mlp.2': [13312, 300],
    'gnn.gnns.1.mlp.2': [13312, 300],
    ...
    'gnn.gnns.4.mlp.2': [13312, 300],
  }
  feature_s = [512, 300]  （图级表示）

Step 2: 提取合并模型+Adapter 的中间层特征（target）
  model 前向传播（surgery=True）：
    Layer 0: h = GINConv(h)                    → [13312, 300]
             h = h - surgery_mlps[0](h)        → [13312, 300]  ← Adapter 作用
             target_getter1 捕获 GINConv 输出:   [13312, 300]
             target_getter2 捕获 surgery_mlps[0].2 输出: [13312, 300]
    ...（5层同理）

    graph_rep = mean_pool(h_last, batch)        → [512, 300]
    graph_rep = graph_rep - surgery_mlp(graph_rep)  → [512, 300]

  # 构造目标特征 = GNN层输出 - Adapter输出（即 Adapter 作用后的节点特征）
  intermediate_output_t['gnn.gnns.0.mlp.2'] =
    target_getter1['gnn.gnns.0.mlp.2'] - target_getter2['gnn.surgery_mlps.0.2']
    = [13312, 300] - [13312, 300] = [13312, 300]

Step 3: 计算 TWD 损失（backbone_regularization）
  对每层（共5层）：
    # 转为密集批量格式（padding 到 batch 内最大节点数 max_N）
    b_nodes_fea_s: [512, max_N, 300]  （source，detach）
    b_nodes_fea_t: [512, max_N, 300]  （target，参与梯度）
    b_mask:        [512, max_N]        bool，有效节点掩码
    b_A:           [512, max_N, max_N] 批量邻接矩阵（加自环）

    # 计算余弦距离代价矩阵
    cos_dist: [512, max_N, max_N]

    # Sinkhorn 迭代（max_iter=100）
    wd: [512]  每个图的 Wasserstein 距离

    twd = 0.5 * mean(wd)  → 标量

  loss_reg_backbone = Σ twd（5层求和）  → 标量

Step 4: 计算特征 L1 损失
  feature_loss = L1(feature_s, feature_t)
    = mean(|[512,300] - [512,300]|)  → 标量

Step 5: 反向传播
  loss = loss_reg_backbone + 1.0 * feature_loss
  loss.backward()
  # 只更新 surgery_mlps（NodeAdapter）和 surgery_mlp（GraphAdapter）的参数
  optimizer.step()
```

**训练结束后保存**：
```
checkpoint = {
  'adapter_layer0': surgery_mlps[0].state_dict(),  # Linear(300,30) + Linear(30,300)
  ...
  'adapter_layer4': surgery_mlps[4].state_dict(),
  'adapter_graph':  surgery_mlp.state_dict(),
}
# 保存到 ./results/gin_contextpred/adapters/bbbp_adapters.pth
```

### 3.5 Phase 3：MoE 路由推理数据流

**模型状态**：加载 Phase 1 合并权重 + 8个任务的 Adapter 作为 8 个 Expert（moe=True）

**一个推理 batch 的完整数据流**（batch_size=512，以 BBBP 测试集为例）：

```
输入: batch.x [N_total, 2], batch.edge_index [2, E_total], batch.edge_attr [E_total, 2]

Step 1: 输入嵌入
  x = embedding1(x[:,0]) + embedding2(x[:,1])  → [N_total, 300]

Step 2: 5层 GNN + 节点级 MoE 路由（以第 layer=0 层为例）
  h = GINConv[0](x, edge_index, edge_attr)     → [N_total, 300]

  # 8个 Expert 各自计算残差
  Expert_0(h): [N_total, 300]  ← BBBP 任务的 NodeAdapter（index=4，bbbp是第5个任务）
  Expert_1(h): [N_total, 300]  ← tox21 任务的 NodeAdapter
  ...
  Expert_7(h): [N_total, 300]  ← muv 任务的 NodeAdapter

  # 转为密集格式（用于 TWD 计算）
  moe_dense_list[i]: [512, max_N, 300]  （i=0..7）

  # 构建邻接矩阵
  b_A: [512, max_N, max_N]

  # TWD 路由：计算参考 Expert（index=args.index，对应当前任务）与各 Expert 的距离
  # 对 BBBP（index=4），参考 Expert 是 Expert_4（BBBP 自己的 Adapter）
  wd_0 = TWD(Expert_4_dense, Expert_0_dense, b_A)  → [512]  （每图一个距离值）
  wd_1 = TWD(Expert_4_dense, Expert_1_dense, b_A)  → [512]
  ...
  wd_7 = TWD(Expert_4_dense, Expert_7_dense, b_A)  → [512]

  # 路由分数（取负：距离越小→分数越高）
  moe_score: [512, 8]  = stack([-wd_0, -wd_1, ..., -wd_7], dim=1)

  # 扩展到节点级
  moe_score = moe_score[batch, :]  → [N_total, 8]

  # Top-8 Softmax（topk=8，即选全部 Expert）
  topk_values: [N_total, 8]  经 softmax(·/0.03) 归一化

  # 加权融合
  moe_rep: [N_total, 8, 300]  = stack(Expert_i(h), dim=1)
  moe_rep = moe_rep * moe_score.unsqueeze(-1)  → [N_total, 8, 300]
  h_moe = moe_rep.sum(dim=1)                   → [N_total, 300]

  # 残差减法
  h = h - h_moe                                → [N_total, 300]

  # BatchNorm + Dropout
  h = BatchNorm(h)                             → [N_total, 300]
  h = Dropout(ReLU(h))                         → [N_total, 300]

  （Layer 1~4 同理，共 5 层）

Step 3: 图池化
  node_rep = h_list[-1]                        → [N_total, 300]
  graph_rep = mean_pool(node_rep, batch)        → [512, 300]

Step 4: 图级 MoE 路由
  # 8个图级 Expert 各自计算
  Expert_i(graph_rep): [512, 300]  （i=0..7）

  # 余弦相似度路由（图级无拓扑结构，用余弦相似度代替 TWD）
  score_i = cosine_similarity(Expert_i(g), Expert_4(g), dim=1)  → [512]
  moe_score: [512, 8]

  # Top-8 Softmax
  topk_values: [512, 8]  经 softmax(·/0.03) 归一化

  # 加权融合
  graph_rep_moe = Σ(Expert_i(g) * score_i)    → [512, 300]
  graph_rep = graph_rep - graph_rep_moe         → [512, 300]

Step 5: 分类预测
  logits = graph_pred_linear(graph_rep)         → [512, 1]  （BBBP num_tasks=1）
  scores = sigmoid(logits)                      → [512, 1]  （概率值 0~1）

输出:
  y_true:  [204, 1]  （测试集所有样本的真实标签）
  y_scores:[204, 1]  （预测概率）
  ROC-AUC: 标量（%），衡量二分类性能
```

### 3.6 数据流维度总览（BBBP，batch_size=512）

| 阶段 | 张量名 | 形状 | 含义 |
|------|--------|------|------|
| 原始输入 | `batch.x` | [~13312, 2] | 批内所有原子的类型+手性特征 |
| 原始输入 | `batch.edge_index` | [2, ~28672] | 批内所有化学键（双向） |
| 原始输入 | `batch.edge_attr` | [~28672, 2] | 批内所有键的类型+方向特征 |
| 嵌入后 | `x` | [~13312, 300] | 原子初始嵌入 |
| GNN 每层输出 | `h` | [~13312, 300] | 节点表示（消息传递后） |
| MoE Expert 输出 | `moe_list[i]` | [~13312, 300] | 第 i 个 Expert 的残差 |
| MoE 密集格式 | `moe_dense_list[i]` | [512, max_N, 300] | 用于 TWD 计算的 padding 格式 |
| 邻接矩阵 | `b_A` | [512, max_N, max_N] | 批量邻接矩阵（含自环） |
| TWD 距离 | `wd` | [512] | 每个图的 Wasserstein 距离 |
| 节点路由分数 | `moe_score` | [~13312, 8] | 每个节点对应图的路由权重 |
| 节点融合残差 | `h_moe` | [~13312, 300] | 加权融合后的节点残差 |
| 图表示 | `graph_rep` | [512, 300] | 池化后的图级表示 |
| 图路由分数 | `moe_score` | [512, 8] | 图级余弦相似度路由权重 |
| 图融合残差 | `graph_rep_moe` | [512, 300] | 图级加权融合残差 |
| 最终预测 | `logits` | [512, 1] | 分类 logit |
| 最终预测 | `scores` | [512, 1] | sigmoid 概率（越接近1越可能穿透血脑屏障） |
