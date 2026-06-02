import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def compute_motif_distribution(edge_index, num_nodes, eps=1e-8):
    """
    Compute motif frequency distribution for a graph.
    Motifs: [edges, triangles, wedges(3-star), 4-cycles, 4-cliques]
    Returns: normalized probability vector [5]
    """
    with torch.no_grad():
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        G = to_networkx(data, to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))

        # 1. Edge count
        n_edges = G.number_of_edges()

        # 2. Triangles: sum of triangles per node / 3
        tri_dict = nx.triangles(G)
        n_triangles = sum(tri_dict.values()) // 3

        # 3. Wedges (3-star / path of length 2): sum of C(deg,2)
        n_wedges = sum(d * (d - 1) // 2 for _, d in G.degree())

        # 4. 4-cycles
        A = nx.to_numpy_array(G)
        import numpy as np
        A2 = A @ A
        A4 = A2 @ A2
        # number of 4-cycles = (tr(A^4) - 4*m*(2*m-1) - 2*sum(d_i*(d_i-1))) / 8
        # simplified: use tr(A^4) - correction
        n_4cycles = max(0, int((np.trace(A4) - 4 * n_edges * (2 * n_edges - 1) - 2 * sum(d * (d - 1) for _, d in G.degree())) // 8))

        # 5. 4-cliques
        n_4cliques = sum(1 for c in nx.enumerate_all_cliques(G) if len(c) == 4)

        counts = torch.tensor(
            [n_edges, n_triangles, n_wedges, n_4cycles, n_4cliques],
            dtype=torch.float32
        )
        counts = counts + eps
        return counts / counts.sum()


class MotifMoERouter(torch.nn.Module):
    """
    Motif-Distribution MoE Router.
    - Phase 2: call add_task_motif(task_id, loader) after adapter training.
    - Phase 3: call route(edge_index, num_nodes) to get MoE weights via JSD.
    - OOD: if min JSD > gamma, register new task anchor.
    """

    def __init__(self, temperature=0.1, novelty_threshold=0.5):
        super().__init__()
        self.tau = temperature
        self.gamma = novelty_threshold
        # {task_id -> motif distribution tensor [5]}
        self.motif_memory = {}

    def add_task_motif(self, task_id: int, loader):
        """Compute and store average motif distribution for a task's dataset."""
        dists = []
        with torch.no_grad():
            for batch in loader:
                # iterate over individual graphs in the batch
                ptr = batch.ptr if hasattr(batch, 'ptr') and batch.ptr is not None else None
                if ptr is not None:
                    for i in range(len(ptr) - 1):
                        node_start, node_end = ptr[i].item(), ptr[i + 1].item()
                        mask = (batch.edge_index[0] >= node_start) & (batch.edge_index[0] < node_end)
                        ei = batch.edge_index[:, mask] - node_start
                        n = node_end - node_start
                        if n > 1 and ei.shape[1] > 0:
                            dists.append(compute_motif_distribution(ei, n))
                else:
                    if batch.num_nodes > 1 and batch.edge_index.shape[1] > 0:
                        dists.append(compute_motif_distribution(batch.edge_index, batch.num_nodes))
        if dists:
            self.motif_memory[task_id] = torch.stack(dists).mean(0)

    def _jsd(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Jensen-Shannon Divergence between two distributions."""
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum()
        kl_qm = (q * (q / m).log()).sum()
        return 0.5 * kl_pm + 0.5 * kl_qm

    def route(self, edge_index: torch.Tensor, num_nodes: int, temperature: float = None):
        """
        Compute routing weights over K tasks.
        Returns dict {task_id: weight}.
        """
        if not self.motif_memory:
            raise ValueError("Motif memory is empty. Call add_task_motif() first.")
        tau = temperature if temperature is not None else self.tau

        with torch.no_grad():
            h_new = compute_motif_distribution(edge_index, num_nodes)
            task_ids = list(self.motif_memory.keys())
            jsds = torch.stack([self._jsd(h_new, self.motif_memory[tid]) for tid in task_ids])
            weights = F.softmax(-jsds / tau, dim=0)

            min_jsd = jsds.min().item()
            if min_jsd > self.gamma:
                new_id = max(task_ids) + 1
                print(f"检测到严重的拓扑漂移 (OOD)，注册新任务锚点 task_id={new_id}")
                self.motif_memory[new_id] = h_new

        return {tid: weights[i] for i, tid in enumerate(task_ids)}

    def get_task_ids(self):
        return list(self.motif_memory.keys())
