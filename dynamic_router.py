import torch
import torch.nn.functional as F


def sinkhorn(C, num_iters=20, eps=0.05):
    """
    Sinkhorn OT on cost matrix C [N, p] with uniform marginals.
    Returns scalar transport cost. Runs under no_grad.
    """
    N, p = C.shape
    # Convert cost to log-domain kernel
    log_K = -C / eps
    log_u = torch.zeros(N, device=C.device)
    log_v = torch.zeros(p, device=C.device)
    log_a = -torch.log(torch.tensor(N, dtype=C.dtype, device=C.device))
    log_b = -torch.log(torch.tensor(p, dtype=C.dtype, device=C.device))
    for _ in range(num_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    T = log_T.exp()
    return (T * C).sum()


class PromptMoERouter(torch.nn.Module):
    """
    Prompt-based Continual Router.
    - Phase 2: call add_task_prompt(task_id, P_k) after adapter training.
    - Phase 3: call route(H_new) to get MoE weights via Mini-TWD (Sinkhorn).
    - Continual: call maybe_expand(H_new) to detect OOD and expand prompt memory.
    """

    def __init__(self, num_prompts=5, temperature=0.1, novelty_threshold=0.5, sinkhorn_iters=20, sinkhorn_eps=0.05):
        super().__init__()
        self.p = num_prompts
        self.tau = temperature
        self.gamma = novelty_threshold
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        # {task_id (int) -> Tensor [p, d]}
        self.prompt_memory = {}

    def add_task_prompt(self, task_id: int, P_k: torch.Tensor):
        """Store trained prompt P_k [p, d] for task_id."""
        self.prompt_memory[task_id] = P_k.detach()

    def _mini_twd(self, H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Compute Mini-TWD between H [N, d] and P [p, d].
        Cost matrix: pairwise cosine distance. Returns scalar.
        """
        H_norm = F.normalize(H, dim=1)
        P_norm = F.normalize(P, dim=1)
        # cosine distance = 1 - cosine_similarity
        C = 1.0 - H_norm @ P_norm.t()  # [N, p]
        return sinkhorn(C, num_iters=self.sinkhorn_iters, eps=self.sinkhorn_eps)

    def _all_distances(self, H: torch.Tensor):
        """Return list of (task_id, distance) for all prompts."""
        results = []
        with torch.no_grad():
            for tid, P in self.prompt_memory.items():
                d = self._mini_twd(H, P.to(H.device))
                results.append((tid, d))
        return results

    def route(self, H: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """
        Compute routing weights over K tasks.
        H: node embeddings [N, d]
        Returns weights [K] ordered by task_id insertion order.
        """
        if not self.prompt_memory:
            raise ValueError("Prompt memory is empty. Call add_task_prompt() first.")
        tau = temperature if temperature is not None else self.tau
        dists = self._all_distances(H)
        D = torch.stack([d for _, d in dists])  # [K]
        weights = F.softmax(-D / tau, dim=0)
        return weights

    def get_task_ids(self):
        return list(self.prompt_memory.keys())

    def maybe_expand(self, task_id_new: int, H: torch.Tensor, emb_dim: int) -> bool:
        """
        OOD detection: if min distance to all existing prompts > gamma,
        initialize a new prompt for task_id_new and add to memory.
        Returns True if expansion happened.
        """
        if not self.prompt_memory:
            P_new = torch.nn.init.xavier_uniform_(
                torch.empty(self.p, emb_dim, device=H.device)
            )
            self.prompt_memory[task_id_new] = P_new.detach()
            return True

        dists = self._all_distances(H)
        min_dist = min(d.item() for _, d in dists)
        if min_dist > self.gamma:
            P_new = torch.nn.init.xavier_uniform_(
                torch.empty(self.p, emb_dim, device=H.device)
            )
            self.prompt_memory[task_id_new] = P_new.detach()
            return True
        return False
