### **Context & Objective (背景与目标)**
You are an expert PyTorch engineer and AI researcher. We are refactoring the **Phase 3 (MoE Router)** and the **Phase 2 (Adapter Training)** of the "G-Merging" framework for Graph Neural Networks. 
The original Phase 3 uses a global Topology-aware Wasserstein Distance (TWD) which is computationally expensive ($O(N^3)$) and fails in continual learning. We previously tried compressing graphs to 1D vectors, but it caused information collapse. 

**Your task is to implement a "Prompt-based Continual Router" (基于图提示学习的连续路由).**
Instead of storing historical graphs or 1D vectors, we will train a small set of "Virtual Prompt Nodes" ($p$ nodes) for each task. During inference, the MoE router will compute a **Mini-TWD** between the input graph and these Prompts.

---

### **Implementation Steps & Requirements (具体修改步骤)**

#### **Step 1: Modify Phase 2 - Train Task-Specific Prompts**
**Target:** The module where `Task-specific Adapters` are trained.
**Logic:**
1. For each task $k$, initialize a learnable Prompt matrix $P_k \in \mathbb{R}^{p \times d}$ (default $p=5$, $d$ is the embedding dimension). Recommend using `nn.Parameter`.
2. During the task's Adapter training, this $P_k$ should be globally connected to all nodes in the task's training subgraphs (acting as virtual nodes).
3. The Prompt $P_k$ must be optimized alongside the Adapter's parameters using the same loss function, so it can absorb the task's structural and feature distribution.
4. Save the trained $P_k$ into a central `Prompt Memory Dictionary` (`{task_id: P_k_tensor}`).

#### **Step 2: Refactor Phase 3 - Implement Mini-TWD Router**
**Target:** The `EvolutionAwareRouter` or `MoERouter` class.
**Logic:**
Remove the old $O(N^3)$ TWD calculation or 1D cosine similarity logic. Replace it with the **Mini-TWD**:
1. **Input:** A new graph's node embeddings $H_{new} \in \mathbb{R}^{N \times d}$.
2. **Retrieve:** Fetch the $K$ sets of Prompts from the memory: $[P_1, P_2, \dots, P_K]$, where each $P_k \in \mathbb{R}^{p \times d}$.
3. **Mini-Cost Matrix:** For each task $k$, compute the pairwise cosine distance cost matrix $C_k \in \mathbb{R}^{N \times p}$ between $H_{new}$ and $P_k$.
4. **Optimal Transport:** Run a lightweight Sinkhorn iteration (or Earth Mover's Distance) on $C_k$ to get the transport cost $D_k$. *(Note: The marginals are uniform, i.e., $1/N$ for $H_{new}$ and $1/p$ for $P_k$)*.
5. **Routing Weights:** Compute the MoE weights using softmax over the negative distances: $\alpha_k = \text{Softmax}(-D_k / \tau)$, where $\tau$ is a temperature hyperparameter.

#### **Step 3: Implement Continual Prompt Expansion (OOD Detection)**
**Target:** The method handling streaming/new graph data.
**Logic:**
Instead of overwriting existing prompts when graph topology drifts:
1. When a new delta graph $\Delta G$ arrives, compute its Mini-TWD distances $D_k$ to all existing $K$ Prompts.
2. **Threshold Check:** If $\min(D_k) > \gamma$ (where $\gamma$ is a pre-defined novelty threshold), it means this graph is Out-of-Distribution (OOD) or has undergone severe concept drift.
3. **Expansion:** Do NOT overwrite old Prompts. Instead, initialize a new Prompt $P_{K+1}$, assign it to the new topology, and add it to the memory dictionary.

---

### **Coding Guidelines for this PR**
* **PyTorch Geometric (PyG):** Ensure compatibility with PyG's `Data` and `Batch` objects when handling $H_{new}$.
* **Efficiency:** Vectorize the Cost Matrix $C_k$ calculation across the $K$ tasks using `torch.cdist` or batch matrix multiplication.
* **Sinkhorn:** Please write a clean, robust, and detached (`torch.no_grad()`) Sinkhorn function for Phase 3, as it is a training-free inference routing step.
* Please provide the complete Python code for the refactored `PromptMoERouter` class, and show a brief pseudo-code snippet of how Step 1 (Prompt Training) integrates with the Adapter training loop.

**Please start by writing the `PromptMoERouter` module.**