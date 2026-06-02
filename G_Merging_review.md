# G-Merging 代码

## 一、三个 Phase 的代码实现文件位置

G-Merging 将多任务图模型合并分为三个阶段：

| Phase | 名称 | 核心文件 | 关键函数/位置 |
|-------|------|----------|--------------|
| Phase 1 | 粗粒度合并（Coarse Merging） | `G_Merging.py` | `test_one_dataset()` 第 243–269 行；`task_vectors.py` |
| Phase 2 | Adapter 训练（Adapter Training） | `G_Merging.py` | `train_one_adapters()` 第 297–414 行；`MWD/gtot_tuning.py` |
| Phase 3 | MoE 路由推理（MoE Routing Inference） | `G_Merging.py` | `test_one_dataset()` 第 277–293 行；`model.py` GNN.forward() |

### 各 Phase 简述

- **Phase 1**：使用 Task Arithmetic，将所有任务的 task vector 求和后乘以缩放系数，叠加到预训练模型权重上，得到粗粒度合并模型。
- **Phase 2**：在粗粒度合并模型基础上，为每个任务独立训练轻量级 Adapter（NodeAdapter + GraphAdapter），训练目标是用 TWD（拓扑感知 Wasserstein 距离）损失使 Adapter 输出逼近该任务的微调模型中间层特征。
- **Phase 3**：推理时，将所有任务的 Adapter 加载为 MoE 的多个 Expert，通过 TWD 计算各 Expert 与参考 Expert 的距离作为路由分数，Top-K Softmax 加权融合，实现无需训练的拓扑感知动态路由。

```python
输入 x
 ↓
Embedding
 ↓
[GNN layer]
 ↓
Surgery:        h ← h - f(h)
 ↓
MoE Surgery:    h ← h - Σ (gate_i * expert_i(h))
 ↓
BN + ReLU
 ↓
JK 聚合
 ↓
输出 node embedding
```
## 二、NodeAdapter 与 GraphAdapter 的具体代码

> 代码位置：`model.py`

在 G-Merging 中，Adapter 分为两类：
- **NodeAdapter**（节点级）：`surgery_mlps`，作用于 GNN 每一层的节点表示
- **GraphAdapter**（图级）：`surgery_mlp`，作用于图池化后的图表示

### 2.1 NodeAdapter 定义（model.py 第 256–271 行）

```python
# GNN.__init__ 中，当 surgery=True 时构建 NodeAdapter
if self.surgery:
    print('rank:' + str(rank))
    self.surgery_mlps = torch.nn.ModuleList()  # 每层一个 Adapter

    for layer in range(num_layer):
        # 每个 NodeAdapter 是一个低秩瓶颈 MLP：300 -> rank -> 300
        surgery_mlp = torch.nn.Sequential(
            torch.nn.Linear(300, rank, bias=False),  # 降维到 rank（默认30）
            torch.nn.ReLU(),                          # 非线性激活
            torch.nn.Linear(rank, 300, bias=False)   # 升维回 300
        )
        # 初始化：第一层 Kaiming 均匀初始化，第二层全零初始化
        # 全零初始化保证训练初期 Adapter 输出为零，不破坏合并模型
        torch.nn.init.kaiming_uniform_(surgery_mlp[0].weight, a=math.sqrt(5))
        torch.nn.init.zeros_(surgery_mlp[2].weight)

        self.surgery_mlps.append(surgery_mlp)
```

### 2.2 NodeAdapter 前向传播（model.py 第 328–329 行）

```python
# GNN.forward 中，每层 GNN 消息传递后应用 NodeAdapter
if self.surgery:
    # 残差减法：h = h - Adapter(h)
    # 含义：Adapter 学习"需要从合并模型中减去的任务无关成分"
    # 等价于让 h 保留任务特定信息，去除跨任务共享噪声
    h = h - self.surgery_mlps[layer](h)
```

### 2.3 GraphAdapter 定义（model.py 第 491–498 行）

```python
# GNN_graphpred.__init__ 中，当 surgery=True 时构建 GraphAdapter
if self.surgery:
    # GraphAdapter 作用于图池化后的图级表示（300维）
    self.surgery_mlp = torch.nn.Sequential(
        torch.nn.Linear(300, args.rank, bias=False),  # 降维
        torch.nn.ReLU(),
        torch.nn.Linear(args.rank, 300, bias=False)   # 升维
    )
    # 同样使用全零初始化第二层，保证初始无影响
    torch.nn.init.kaiming_uniform_(self.surgery_mlp[0].weight, a=math.sqrt(5))
    torch.nn.init.zeros_(self.surgery_mlp[2].weight)
```

### 2.4 GraphAdapter 前向传播（model.py 第 532–533 行）

```python
# GNN_graphpred.forward 中，图池化后应用 GraphAdapter
if self.surgery:
    # 同样是残差减法：graph_representation = graph_rep - Adapter(graph_rep)
    graph_representation = graph_representation - self.surgery_mlp(graph_representation)
```

### 2.5 Adapter 训练时的优化器设置（G_Merging.py 第 377–384 行）

```python
# Phase 2 训练时，只优化 Adapter 参数，冻结 GNN 主干
optimizer = torch.optim.Adam(
    [
        # GraphAdapter 参数，使用 lr_graph 学习率
        {"params": model.surgery_mlp.parameters(), "lr": args.lr_graph},
        # NodeAdapter 参数（所有层），使用 lr_node 学习率
        {"params": model.gnn.surgery_mlps.parameters(), "lr": args.lr_node},
    ],
    betas=(0.9, 0.999),
    weight_decay=0.
)
```

### 2.6 Adapter 保存（G_Merging.py 第 401–411 行）

```python
# 训练完成后，将每层 NodeAdapter 和 GraphAdapter 分别保存
checkpoint = {
    'adapter_layer0': model.gnn.surgery_mlps[0].state_dict(),
    'adapter_layer1': model.gnn.surgery_mlps[1].state_dict(),
    'adapter_layer2': model.gnn.surgery_mlps[2].state_dict(),
    'adapter_layer3': model.gnn.surgery_mlps[3].state_dict(),
    'adapter_layer4': model.gnn.surgery_mlps[4].state_dict(),
    'adapter_graph': model.surgery_mlp.state_dict(),  # GraphAdapter
}
torch.save(checkpoint, f"./results/{args.gnn_type}_{args.pretrain_strategy}/adapters/{args.dataset}_adapters.pth")
```

---

## 三、TWD（拓扑感知 Wasserstein 距离）的具体代码实现

TWD 由两个文件共同实现：
- `MWD/gtot.py`：实现 Sinkhorn 算法求解最优传输（OT）
- `MWD/gtot_tuning.py`：封装 GTOTRegularization，将 TWD 作为训练损失

### 3.1 GTOT 类（MWD/gtot.py）

```python
class GTOT(nn.Module):
    """
    图拓扑感知最优传输（Graph Topology-aware Optimal Transport）
    使用 Sinkhorn 迭代算法求解带图拓扑约束的 Wasserstein 距离
    """
    def __init__(self, eps=0.1, thresh=0.1, max_iter=100, reduction='none'):
        super(GTOT, self).__init__()
        self.eps = eps          # Sinkhorn 正则化系数（熵正则）
        self.max_iter = max_iter  # 最大迭代次数
        self.reduction = reduction
        self.thresh = thresh    # 收敛阈值
        self.mask_matrix = None

    def forward(self, x, y, C=None, A=None, mask=None):
        """
        计算 x 和 y 之间的（带拓扑约束的）Wasserstein 距离
        Args:
            x: 源节点特征 [bs, N, d]
            y: 目标节点特征 [bs, N, d]
            C: 代价矩阵（若为 None 则自动计算余弦距离）
            A: 图邻接矩阵（拓扑掩码），用于限制传输只在邻居间发生
            mask: 有效节点掩码（处理 padding）
        Returns:
            cost: Wasserstein 距离
            pi:   最优传输计划矩阵
            C:    代价矩阵
        """
        if C is None:
            C = self._cost_matrix(x, y)  # 默认 L2 代价
            C = C / C.max()
        if A is not None:
            # 用邻接矩阵 A 对代价矩阵做掩码：非邻居节点间代价置零
            # 这是"拓扑感知"的核心：只允许图结构相邻的节点间传输
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C

        # 初始化均匀边际分布（每个节点权重相等）
        if mask is None:
            mu = torch.empty(...).fill_(1.0 / N_s)  # 源分布
            nu = torch.empty(...).fill_(1.0 / N_t)  # 目标分布
        else:
            mu, nu = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Sinkhorn 迭代：交替更新对偶变量 u, v
        for i in range(self.max_iter):
            u1 = u
            # 对数域更新 u（数值稳定）
            u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
            # 对数域更新 v
            v = self.eps * (torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).max()
            if err.item() < self.thresh:
                break  # 收敛则提前退出

        # 计算最优传输计划 pi 和 Wasserstein 距离
        pi = self.exp_M(C, u, v, A=A)          # 传输计划矩阵
        cost = torch.sum(pi * C, dim=(-2, -1))  # W 距离 = sum(pi * C)
        return cost, pi, C

    def exp_M(self, C, u, v, A=None):
        """
        计算 Sinkhorn 核矩阵 exp(M)，M = (-C + u + v^T) / eps
        若有邻接矩阵 A，则将非邻居位置置零（拓扑约束）
        """
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                # masked_fill：非邻居位置（A=0）填充为 0
                S = torch.exp(self.M(C, u, v)).masked_fill(mask=(1-a).to(torch.bool), value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask=(1-A).to(torch.bool), value=0)
            return S
        return torch.exp(self.M(C, u, v))

    def cost_matrix_batch_torch(self, x, y, mask=None):
        """
        批量计算余弦距离矩阵
        x: [bs, d, m]，y: [bs, d, m]
        返回: [bs, n, m] 余弦距离（1 - cosine_similarity）
        """
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)  # L2 归一化
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # 余弦相似度
        cos_dis = 1 - cos_dis  # 转为距离（越小越相似）
        return cos_dis.transpose(2, 1)
```

### 3.2 GTOTRegularization 类（MWD/gtot_tuning.py）

```python
class GTOTRegularization(nn.Module):
    """
    将 GTOT 距离封装为训练正则化损失（TWD Loss）
    在 Phase 2 Adapter 训练中，用于约束 Adapter 输出与微调模型中间层特征的拓扑结构一致性
    """
    def __init__(self, order=1, args=None):
        super(GTOTRegularization, self).__init__()
        self.Gtot = GTOT(eps=0.1, thresh=0.1, max_iter=100, reduction=None)
        self.order = order  # 邻接矩阵的幂次 A^order，控制拓扑感知范围

    def got_dist(self, f_s, f_t, A=None, mask=None):
        """
        计算两组节点特征之间的 TWD（拓扑感知 Wasserstein 距离）
        Args:
            f_s: 源特征（来自微调模型，detach 不参与梯度）[bs, N, d]
            f_t: 目标特征（来自带 Adapter 的合并模型）[bs, N, d]
            A:   批量邻接矩阵 [bs, N, N]
            mask: 有效节点掩码（处理不等长图的 padding）
        Returns:
            twd: 标量 TWD 损失
            wd:  每个图的 Wasserstein 距离向量
        """
        # 1. 计算余弦距离矩阵作为 OT 代价矩阵
        cos_distance = self.Gtot.cost_matrix_batch_torch(
            f_s.transpose(2, 1), f_t.transpose(2, 1), mask=mask
        )
        cos_dist = cos_distance.transpose(1, 2)

        # 2. 根据 order 计算 A^order 作为拓扑掩码
        # order=1: 直接使用邻接矩阵（只允许直接邻居间传输）
        # order>1: A^order 扩展到 k 跳邻居
        if self.order == 1:
            A = A  # 直接使用原始邻接矩阵
        elif self.order > 1:
            A0 = A
            for i in range(self.order - 1):
                A = A.bmm(A0)  # 矩阵乘法扩展邻居范围
            A = torch.sign(A)  # 二值化

        # 3. 将余弦掩码与拓扑掩码结合
        if A is not None:
            A = self.Gtot.mask_matrix * A  # 同时满足：有效节点 AND 拓扑邻居

        # 4. 调用 GTOT 计算带拓扑约束的 Wasserstein 距离
        wd, P, C = self.Gtot(x=f_s, y=f_t, A=A, C=cos_dist, mask=mask)
        twd = 0.5 * torch.mean(wd)  # 对 batch 内所有图取均值
        return twd, wd

    def forward(self, layer_outputs_source, layer_outputs_target, *argv):
        """
        对所有 GNN 层的中间特征计算 TWD 损失并求和
        Args:
            layer_outputs_source: 微调模型各层输出（OrderedDict）
            layer_outputs_target: 合并模型+Adapter 各层输出（OrderedDict）
            *argv: 图数据（x, edge_index, edge_attr, batch）
        """
        x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        output = 0.0

        for i, (fm_src, fm_tgt) in enumerate(
            zip(layer_outputs_source.values(), layer_outputs_target.values())
        ):
            # 将稀疏节点特征转为密集批量格式（padding 到最大节点数）
            b_nodes_fea_s, b_mask_s = PyG_utils.to_dense_batch(x=fm_src.detach(), batch=batch)
            b_nodes_fea_t, b_mask_t = PyG_utils.to_dense_batch(x=fm_tgt, batch=batch)

            # 构建批量邻接矩阵（加自环）
            edge_index, _ = PyG_utils.add_remaining_self_loops(edge_index, num_nodes=fm_tgt.size(0))
            b_A = PyG_utils.to_dense_adj(edge_index, batch=batch)

            # 计算该层的 TWD 损失（源特征 detach，只优化目标侧 Adapter）
            distance, _ = self.got_dist(
                f_s=b_nodes_fea_s.detach(),  # 微调模型特征，不参与梯度
                f_t=b_nodes_fea_t,            # Adapter 输出特征，参与梯度
                A=b_A,
                mask=b_mask_t
            )
            output = output + torch.sum(distance)  # 累加各层损失

        return output  # 总 TWD 损失
```

### 3.3 TWD 在训练循环中的使用（G_Merging.py 第 135–189 行）

```python
def train(args, model, finetune_model, loader, optimizer, loss_func,
          finetune_getter, target_getter1, target_getter2, backbone_regularization):
    """
    Phase 2 的单 epoch 训练函数
    同时优化两个损失：
      1. TWD 损失（backbone_regularization）：拓扑感知特征对齐
      2. 特征 L1 损失（feature_loss）：图级表示对齐
    """
    for batch in loader:
        # 获取微调模型各层中间特征（source，不参与梯度）
        intermediate_output_s, feature_s = finetune_getter(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        # 获取合并模型各层中间特征（target1：GNN层输出）
        intermediate_output_t1, feature_t = target_getter1(...)
        # 获取 Adapter 输出（target2：surgery_mlps 输出）
        intermediate_output_t2, feature_t = target_getter2(...)

        # 构造目标特征 = GNN层输出 - Adapter输出（即 Adapter 作用后的结果）
        intermediate_output_t = OrderedDict()
        for i in range(args.num_layer):
            intermediate_output_t[f'gnn.gnns.{i}.mlp.2'] = (
                intermediate_output_t1[f'gnn.gnns.{i}.mlp.2']
                - intermediate_output_t2[f'gnn.surgery_mlps.{i}.2']
            )

        # TWD 损失：约束 Adapter 后的节点特征与微调模型特征的拓扑结构一致
        loss_reg_backbone = backbone_regularization(
            intermediate_output_s, intermediate_output_t, batch
        )

        # 特征 L1 损失：约束图级表示与微调模型一致
        feature_loss = loss_func(feature_s, feature_t)

        # 总损失 = TWD损失 + alpha * 特征损失
        loss = loss_reg_backbone + args.alpha * feature_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 四、MoE Router 的具体代码实现

MoE Router 在 Phase 3 推理时使用，将各任务 Adapter 作为 Expert，通过 TWD 计算路由分数，动态选择最相关的 Expert 组合。

### 4.1 MoE Expert 定义（model.py 第 274–290 行，节点级）

```python
# GNN.__init__ 中，当 moe=True 时构建节点级 MoE Expert
if self.moe:
    self.surgery_moe_layers = torch.nn.ModuleList()  # 每层一个 Expert 列表
    self.TWD = GTOTRegularization(order=order)        # TWD 路由器

    for layer in range(num_layer):
        surgery_moe = torch.nn.ModuleList()
        for i in range(num_experts):  # 默认 8 个 Expert（对应 8 个任务）
            # 每个 Expert 结构与 NodeAdapter 相同：低秩瓶颈 MLP
            surgery_mlp = torch.nn.Sequential(
                torch.nn.Linear(300, rank, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(rank, 300, bias=False)
            )
            torch.nn.init.kaiming_uniform_(surgery_mlp[0].weight, a=math.sqrt(5))
            torch.nn.init.zeros_(surgery_mlp[2].weight)
            surgery_moe.append(surgery_mlp)
        self.surgery_moe_layers.append(surgery_moe)  # [num_layer][num_experts]
```

### 4.2 图级 MoE Expert 定义（model.py 第 501–511 行）

```python
# GNN_graphpred.__init__ 中，当 moe=True 时构建图级 MoE Expert
if self.moe:
    self.surgery_moe = torch.nn.ModuleList()
    for i in range(args.num_experts):
        surgery_mlp = torch.nn.Sequential(
            torch.nn.Linear(300, args.rank, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(args.rank, 300, bias=False)
        )
        torch.nn.init.kaiming_uniform_(surgery_mlp[0].weight, a=math.sqrt(5))
        torch.nn.init.zeros_(surgery_mlp[2].weight)
        self.surgery_moe.append(surgery_mlp)
```

### 4.3 节点级 MoE Router 前向传播（model.py 第 331–379 行）

```python
# GNN.forward 中，每层 GNN 消息传递后执行 MoE 路由
if self.moe:
    moe_list = []       # 各 Expert 的输出（节点特征残差）
    moe_dense_list = [] # 各 Expert 输出的密集批量格式
    wds = []            # 各 Expert 与参考 Expert 的 TWD 距离

    # Step 1: 计算所有 Expert 的输出
    for i in range(len(self.surgery_moe_layers[layer])):
        nodes_fea = self.surgery_moe_layers[layer][i](h)  # Expert_i(h)
        moe_list.append(nodes_fea)
        # 转为密集格式（用于 TWD 计算）
        nodes_fea_dense, mask = PyG_utils.to_dense_batch(x=moe_list[i], batch=batch)
        moe_dense_list.append(nodes_fea_dense)

    # 将所有 Expert 输出堆叠：[num_nodes, num_experts, 300]
    moe_rep = torch.stack(moe_list, dim=1)

    # 构建批量邻接矩阵（加自环）
    edge_index_new, _ = PyG_utils.add_remaining_self_loops(
        edge_index, num_nodes=moe_list[0].size(0)
    )
    b_A = PyG_utils.to_dense_adj(edge_index_new, batch=batch)

    # Step 2: 用 TWD 计算每个 Expert 与参考 Expert（index=0）的距离
    for i in range(len(self.surgery_moe_layers[layer])):
        _, wd = self.TWD.got_dist(
            f_s=moe_dense_list[self.index],  # 参考 Expert（默认第0个）
            f_t=moe_dense_list[i],           # 当前 Expert
            A=b_A,
            mask=mask
        )
        wds.append(-wd)  # 取负值：距离越小 → 分数越高（越相似）

    # Step 3: 构建路由分数矩阵 [bs, num_experts]
    moe_score = torch.stack(wds, dim=1)
    # 将图级分数扩展到节点级：每个节点使用其所属图的路由分数
    moe_score = moe_score[batch, :]  # [num_nodes, num_experts]

    # Step 4: Top-K 选择（默认 topk=8，即选所有 Expert）
    topk_values, topk_indices = torch.topk(moe_score, self.topk, dim=1)
    T = 0.03  # 温度系数（越小分布越尖锐，越接近 hard selection）
    # Softmax 归一化得到 Expert 权重
    topk_values = F.softmax(topk_values / T, dim=1)

    # Step 5: 构建稀疏权重矩阵（非 Top-K 位置为 0）
    mask = torch.zeros_like(moe_score)
    mask.scatter_(1, topk_indices, topk_values)
    moe_score = mask  # [num_nodes, num_experts]

    # Step 6: 加权融合所有 Expert 输出
    moe_score = moe_score.unsqueeze(-1).expand(-1, -1, 300)  # [num_nodes, num_experts, 300]
    moe_rep = moe_rep * moe_score   # 加权
    h_moe = moe_rep.sum(dim=1)      # 求和得到融合后的残差 [num_nodes, 300]

    # Step 7: 残差减法（与 NodeAdapter 一致的接口）
    h = h - h_moe
```

### 4.4 图级 MoE Router 前向传播（model.py 第 535–553 行）

```python
# GNN_graphpred.forward 中，图池化后执行图级 MoE 路由
if self.moe:
    moe_list = []
    score_list = []

    # Step 1: 计算所有图级 Expert 的输出
    for i in range(len(self.surgery_moe)):
        moe_list.append(self.surgery_moe[i](graph_representation))
    moe_rep = torch.stack(moe_list, dim=1)  # [bs, num_experts, 300]

    # Step 2: 图级路由使用余弦相似度（而非 TWD）
    # 注：图级特征已经是全局聚合后的向量，不含图结构信息，故用余弦相似度
    for i in range(len(self.surgery_moe)):
        score_list.append(
            F.cosine_similarity(moe_list[i], moe_list[self.index], dim=1)
        )
    moe_score = torch.stack(score_list, dim=1)  # [bs, num_experts]

    # Step 3: Top-K + Softmax 路由（与节点级相同）
    topk_values, topk_indices = torch.topk(moe_score, self.topk, dim=1)
    T = 0.03
    topk_values = F.softmax(topk_values / T, dim=1)
    mask = torch.zeros_like(moe_score)
    mask.scatter_(1, topk_indices, topk_values)
    moe_score = mask

    # Step 4: 加权融合并残差减法
    moe_score = moe_score.unsqueeze(-1).expand(-1, -1, 300)
    moe_rep = moe_rep * moe_score
    graph_representation_moe = moe_rep.sum(dim=1)
    graph_representation = graph_representation - graph_representation_moe
```

### 4.5 MoE Expert 加载（G_Merging.py 第 277–283 行）

```python
# Phase 3 推理前，将各任务训练好的 Adapter 加载为 MoE 的各个 Expert
list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
for i, d_name in enumerate(list_datasets):
    # 加载第 i 个任务的 Adapter 文件
    model_file = f'./results/{args.gnn_type}_{args.pretrain_strategy}/adapters/{d_name}_adapters.pth'
    dict_moe = torch.load(model_file, map_location='cpu')
    # 将各层 NodeAdapter 加载为第 i 个 Expert
    for layer in range(args.num_layer):
        model.gnn.surgery_moe_layers[layer][i].load_state_dict(dict_moe[f'adapter_layer{layer}'])
    # 将 GraphAdapter 加载为图级第 i 个 Expert
    model.surgery_moe[i].load_state_dict(dict_moe['adapter_graph'])
```

---

## 五、整体流程总结

```
预训练模型
    │
    ▼ Phase 1: Task Arithmetic（task_vectors.py + G_Merging.py）
粗粒度合并模型 = pretrained + λ * Σ(task_vectors)
    │
    ▼ Phase 2: Adapter 训练（G_Merging.py train_one_adapters + MWD/gtot_tuning.py）
    对每个任务独立训练 NodeAdapter + GraphAdapter
    损失 = TWD(合并模型+Adapter 中间层, 微调模型中间层) + α * L1(图级表示)
    │
    ▼ Phase 3: MoE 路由推理（G_Merging.py test_one_dataset + model.py）
    将所有任务 Adapter 加载为 MoE Expert
    节点级路由：TWD 距离 → Top-K Softmax 加权融合
    图级路由：余弦相似度 → Top-K Softmax 加权融合
    最终预测 = graph_pred_linear(合并后图表示)
```

