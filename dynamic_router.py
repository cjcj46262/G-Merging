import torch
import torch.nn.functional as F


class EvolutionAwareRouter(torch.nn.Module):
    """
    Replaces TWD-based MoE routing with topology-aware embedding memory (TEM).
    - Memory construction: after each task's adapter training, call add_task_to_memory()
    - Inference routing: call route() to get per-expert weights
    - Drift calibration: call detect_and_calibrate_drift() when graph topology changes
    """

    def __init__(self, feature_dim=300, reduced_dim=64, num_layers=2, drift_threshold=0.5):
        super().__init__()
        self.L = num_layers
        self.drift_threshold = drift_threshold

        # Fixed random orthogonal projection for dimensionality reduction; not trained
        W = torch.empty(feature_dim, reduced_dim)
        torch.nn.init.orthogonal_(W)
        self.register_buffer('W_proj', W)

        # TEM: {task_id (int) -> anchor vector [reduced_dim]}
        self.tem_buffer = {}

    def _sgc_diffuse(self, x, edge_index, num_nodes):
        """
        Parameter-free SGC diffusion: H = (D^{-0.5} A D^{-0.5})^L * X
        Returns node embeddings of shape [num_nodes, x.size(1)]
        """
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [num_edges]

        h = x
        for _ in range(self.L):
            h_next = torch.zeros_like(h)
            h_next.index_add_(0, row, h[col] * norm.unsqueeze(-1))
            h = h_next
        return h

    def _topology_embedding(self, x, edge_index):
        """Compute graph-level anchor: diffuse then project then mean-pool."""
        num_nodes = x.size(0)
        h = self._sgc_diffuse(x, edge_index, num_nodes)
        h_proj = h @ self.W_proj  # [num_nodes, reduced_dim]
        return h_proj

    def add_task_to_memory(self, task_id: int, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Call after task k's adapter training finishes.
        x: node features [N, feature_dim] (post-embedding, float)
        edge_index: [2, E]
        """
        with torch.no_grad():
            h_proj = self._topology_embedding(x, edge_index)
            anchor = h_proj.mean(dim=0)  # [reduced_dim]
        self.tem_buffer[task_id] = anchor.detach()

    def route(self, x: torch.Tensor, edge_index: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Fast routing at inference time.
        Returns routing_weights: [K] tensor (ordered by task_id insertion order).
        """
        if not self.tem_buffer:
            raise ValueError("TEM buffer is empty. Call add_task_to_memory() first.")

        with torch.no_grad():
            h_proj = self._topology_embedding(x, edge_index)
            query = h_proj.mean(dim=0)  # [reduced_dim]

        task_ids = list(self.tem_buffer.keys())
        anchors = torch.stack([self.tem_buffer[tid] for tid in task_ids])  # [K, reduced_dim]

        sim = F.cosine_similarity(query.unsqueeze(0), anchors)  # [K]
        weights = F.softmax(sim / temperature, dim=0)  # [K]
        return weights  # caller indexes by position matching task_ids order

    def get_task_ids(self):
        return list(self.tem_buffer.keys())

    def detect_and_calibrate_drift(self, task_id: int, x_old: torch.Tensor, edge_index_old: torch.Tensor,
                                   x_new: torch.Tensor, edge_index_new: torch.Tensor,
                                   theta_uni, momentum: float = 0.8):
        """
        DyGRAIN-style drift detection and TEM anchor update.
        theta_uni: callable (x, edge_index) -> node embeddings [N, D]
        """
        if task_id not in self.tem_buffer:
            return

        num_old = x_old.size(0)
        num_new = x_new.size(0)

        # --- Structural Influence (SI) ---
        delta_indicator = torch.zeros(num_new, 1, device=x_new.device)
        delta_indicator[num_old:] = 1.0
        si_h = self._sgc_diffuse(delta_indicator, edge_index_new, num_new)
        si_scores = si_h[:num_old].norm(dim=1)  # [num_old]

        # --- Feature Influence (FI) ---
        with torch.no_grad():
            emb_old = theta_uni(x_old, edge_index_old)
            emb_new = theta_uni(x_new, edge_index_new)[:num_old]
        fi_scores = 1.0 - F.cosine_similarity(emb_old, emb_new)  # [num_old]

        # --- Combined drift score ---
        si_scores = si_scores / (si_scores.max() + 1e-9)
        fi_scores = fi_scores / (fi_scores.max() + 1e-9)
        drift_scores = si_scores + fi_scores

        drifted_mask = drift_scores > self.drift_threshold
        if not drifted_mask.any():
            return

        with torch.no_grad():
            h_new_proj = self._topology_embedding(x_new, edge_index_new)
            # Use only the drifted old-node positions to compute correction
            corrected_anchor = h_new_proj[:num_old][drifted_mask].mean(dim=0)

        old_anchor = self.tem_buffer[task_id]
        self.tem_buffer[task_id] = (momentum * old_anchor + (1 - momentum) * corrected_anchor).detach()
