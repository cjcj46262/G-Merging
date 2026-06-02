如果只针对 **Phase 3（MoE 路由阶段）** 进行魔改，这是整篇工作中最具性价比且最容易跑出惊艳 Baseline 的切入点。原版 G-Merging 在 Phase 3 的核心痛点在于：**基于 TWD 的在线路由计算极度耗时，且完全无法抵抗连续图学习中的“拓扑演化（概念漂移）”**。

我们将摒弃原版庞大的代价矩阵计算，引入 **PDGNNs 的拓扑感知嵌入记忆库 (TEM)** 和 **DyGRAIN 的概念漂移评分 (SI & FI)**，构建一个**演化感知与内存高效的动态路由器 (Evolution-Aware Dynamic Router)**。

以下是仅修改 Phase 3 的详细实施方案、数据流设计与核心代码。

---

### 一、 核心数据流对比 (Data Flow)

**原版 G-Merging Phase 3 数据流：**
输入新图 $G_{new}$ $\rightarrow$ 提取全图特征和邻接矩阵 $\rightarrow$ 与 $K$ 个历史任务的图分布进行最优传输计算 (TWD) $\rightarrow$ 得到 $K$ 个距离 $D_k$ $\rightarrow$ Softmax 转化为路由权重 $\alpha_k$ $\rightarrow$ 组合 Adapters。
*(缺点：每次推理都要算 $K$ 次全图级别的距离，慢且存不下历史图。)*

**改造后的 CFG-Merging Phase 3 数据流：**
1. **记忆构建 (Task 训练结束时)：**
   当前图 $G_k$ $\rightarrow$ 无参数拓扑扩散 (SGC) $\rightarrow$ 降维得到 1D 向量 (Anchor $Z_k$) $\rightarrow$ 存入 TEM 缓冲库。
2. **极速路由 (推理时)：**
   输入新节点/图 $G_{new}$ $\rightarrow$ 无参数拓扑扩散提取局部特征 $Z_{new}$ $\rightarrow$ 计算与 TEM 库中 $K$ 个 Anchor 的欧式/余弦距离 $\rightarrow$ Softmax 得到权重 $\alpha_k$。
3. **漂移校准 (增量拓扑 $\Delta G$ 到来时)：**
   输入新连边 $\Delta G$ $\rightarrow$ DyGRAIN 计算 SI & FI 分数 $\rightarrow$ 筛选出分数 $>$ 阈值的受污染 Anchor $\rightarrow$ 利用新图重新提取这些 Anchor 的 $Z$ 并覆盖 TEM 库中的旧值。

---

### 二、 详细改进方案与数学映射

#### 1. 记忆降维 (TEM) 代替 TWD
利用图卷积的低通滤波特性，将图的结构信息压缩进特征中。
给定图的特征矩阵 $X$ 和归一化邻接矩阵 $\tilde{A}$，我们计算拓扑感知嵌入 (TE)：
$$H^{(L)} = (\tilde{A})^L X W_{proj}$$
其中 $L$ 是扩散步数（如 2 或 3），$W_{proj}$ 是一个固定的、未经训练的随机正交矩阵，仅仅为了把高维特征降维到很小的维度（如 64 维），极其省内存。将一个任务中所有节点的 $H^{(L)}$ 求均值，作为该任务在 TEM 中的 Anchor $Z_k$。

#### 2. DyGRAIN 漂移评分 (SI & FI)
当图中新增了一批节点或连边时，计算图上已有节点的受影响程度 $S_{IP}(v)$。
* **结构影响 (SI)：** 假设 $\Delta V$ 是新加入的节点集合，定义指示向量 $I_{\Delta V}$（新节点位置为 1，旧节点为 0）。通过邻接矩阵扩散计算旧节点受到的结构冲击：$SI = (\tilde{A})^L I_{\Delta V}$。
* **特征影响 (FI)：** 将新图和旧图分别输入给 Phase 1 已经融合好的骨干网络 $\theta_{uni}$，计算旧节点特征的变化量：$FI(v) = 1 - \cos(h_v^{old}, h_v^{new})$。
* **总分：** $S_{IP}(v) = SI(v) + FI(v)$。提取 Top-K 受影响的节点进行 Anchor 刷新。

---

### 三、 PyTorch / PyG 伪代码实现

你可以直接新建一个 `dynamic_router.py`，将以下模块接入 G-Merging 的代码库中。

```python
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian

class EvolutionAwareRouter(torch.nn.Module):
    def __init__(self, feature_dim, reduced_dim=64, num_layers=2, drift_threshold=0.5):
        super().__init__()
        self.L = num_layers
        self.drift_threshold = drift_threshold
        
        # 随机正交投影矩阵 W_proj，冻结不训练，仅用于降维以节省 TEM 内存
        self.register_buffer('W_proj', torch.nn.init.orthogonal_(torch.empty(feature_dim, reduced_dim)))
        
        # TEM 记忆库：存储任务 k 的拓扑感知锚点 (Anchor)
        # 格式: {task_id: anchor_vector_1D}
        self.tem_buffer = {} 

    def _compute_topology_embedding(self, x, edge_index):
        """
        替代 TWD 的核心：计算拓扑感知嵌入 (TE)
        相当于无参数的 SGC (Simple Graph Convolution) 扩散
        """
        # 为了简便，这里省略了 edge_weight 的计算，实际应用中建议添加 self-loop 和 degree normalization
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * 1.0 * deg_inv_sqrt[col] # D^{-0.5} A D^{-0.5}

        # 扩散 L 步
        h = x
        for _ in range(self.L):
            # 稀疏矩阵乘法：\tilde{A} * H
            h_next = torch.zeros_like(h)
            h_next.index_add_(0, row, h[col] * norm.view(-1, 1))
            h = h_next
            
        # 降维：避免 TEM 内存爆炸
        h_reduced = torch.matmul(h, self.W_proj)
        return h_reduced

    def add_task_to_memory(self, task_id, x, edge_index):
        """在每个任务 Adapter 训练结束后调用，将其拓扑印记存入 TEM"""
        te_node_level = self._compute_topology_embedding(x, edge_index)
        # 取该任务所有节点的均值作为该任务的 Anchor
        task_anchor = te_node_level.mean(dim=0)
        self.tem_buffer[task_id] = task_anchor

    def route(self, x_new, edge_index_new, temperature=0.1):
        """
        推理时的路由操作：极速计算，彻底取代原来的全局 TWD
        """
        if not self.tem_buffer:
            raise ValueError("TEM buffer is empty. Add tasks first.")
            
        # 1. 计算当前新图的拓扑嵌入
        te_new = self._compute_topology_embedding(x_new, edge_index_new)
        graph_anchor_new = te_new.mean(dim=0)
        
        # 2. 计算与历史所有任务 Anchor 的距离 (使用余弦相似度)
        task_ids = list(self.tem_buffer.keys())
        anchors = torch.stack([self.tem_buffer[tid] for tid in task_ids]) # Shape: [K, reduced_dim]
        
        # 计算相似度 logits
        sim_logits = F.cosine_similarity(graph_anchor_new.unsqueeze(0), anchors)
        
        # 3. Softmax 转化为路由权重 alpha_k
        routing_weights = F.softmax(sim_logits / temperature, dim=0)
        
        # 返回字典 {task_id: weight}，用于后续对 Adapter 进行加权求和
        return {task_ids[i]: routing_weights[i].item() for i in range(len(task_ids))}

    def detect_and_calibrate_drift(self, task_id, x_old, edge_index_old, x_new, edge_index_new, theta_uni):
        """
        基于 DyGRAIN 的概念漂移检测与校准 (增量图到达时调用)
        """
        # --- 1. 计算结构影响 (Structural Influence, SI) ---
        # 找出新图相对于旧图新增的节点索引 (简化逻辑)
        num_old_nodes = x_old.size(0)
        num_new_nodes = x_new.size(0)
        delta_indicator = torch.zeros(num_new_nodes, 1)
        delta_indicator[num_old_nodes:] = 1.0 # 新节点标记为 1
        
        # SI = 增量指示器在拓扑上的扩散
        si_scores = self._compute_topology_embedding(delta_indicator, edge_index_new) 
        si_scores = si_scores[:num_old_nodes].norm(dim=1) # 只看旧节点受到的冲击
        
        # --- 2. 计算特征影响 (Feature Influence, FI) ---
        with torch.no_grad():
            emb_old = theta_uni(x_old, edge_index_old)
            # 新图输入统一骨干网络，取出旧节点对应的部分
            emb_new_full = theta_uni(x_new, edge_index_new)
            emb_new_for_old_nodes = emb_new_full[:num_old_nodes]
            
        # FI = 特征余弦距离
        fi_scores = 1.0 - F.cosine_similarity(emb_old, emb_new_for_old_nodes)
        
        # --- 3. 综合打分与校准 ---
        # 归一化 SI 和 FI (简化处理)
        si_scores = si_scores / (si_scores.max() + 1e-9)
        fi_scores = fi_scores / (fi_scores.max() + 1e-9)
        
        drift_scores = si_scores + fi_scores
        
        # 找出受影响最严重的旧节点 (漂移分数大于阈值)
        drifted_nodes_mask = drift_scores > self.drift_threshold
        
        if drifted_nodes_mask.any():
            print(f"检测到 Task {task_id} 发生拓扑漂移！正在刷新 TEM...")
            # 利用新图结构重新计算这些受损节点的 TE，并更新 Anchor
            te_new_full = self._compute_topology_embedding(x_new, edge_index_new)
            te_drifted_corrected = te_new_full[:num_old_nodes][drifted_nodes_mask]
            
            # 使用动量更新 (Momentum Update) 刷新该任务的 Anchor
            momentum = 0.8
            new_anchor = te_drifted_corrected.mean(dim=0)
            self.tem_buffer[task_id] = momentum * self.tem_buffer[task_id] + (1 - momentum) * new_anchor
```

### 四、 为什么这套方案在论文中极具说服力？

1. **逻辑闭环完美：** 原论文 Phase 3 是静态图的最优传输，你指出这在 Continual Learning 中不成立（因为 $A$ 矩阵会变）。你用 DyGRAIN 定量衡量了 $A$ 的变化，用 PDGNN 的 TEM 解决了 TWD 计算耗时的问题。
2. **实验效果预期极好：** 原始 TWD 推理需要 $O(N^3)$ 复杂度，你把它降维到了 $O(|E|)$ 的稀疏矩阵乘法。在对比实验（Ablation Study）中，你的 **Inference Latency (推理延迟)** 和 **Memory Usage (显存占用)** 会形成压倒性优势。
3. **即插即用：** 这个 `EvolutionAwareRouter` 类完全不需要参与 Phase 1 和 Phase 2 的梯度反向传播。它是一个独立的轻量级模块，直接替换掉原代码中计算 Wasserstein Distance 的那一段即可，工程实现难度极低。