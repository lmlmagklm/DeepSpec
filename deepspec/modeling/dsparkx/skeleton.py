"""通用专家（General / Skeleton Expert）.

对应设计图中的「通用专家 · 出轮廓」：在进入 K 个小专家并行解码之前，
先由通用专家对共享向量表征 H 做一次粗粒度塑形，产出两类信号：

1. sketch（草图向量）：回投到隐空间，作为每个小专家的额外条件输入，
   让各专家在同一「轮廓」上分工细化；
2. coarse logits（粗分布）：低秩词表投影出的块级轮廓分布，用于
   skeleton CE 监督与训练时的接受率诊断。
"""

from torch import nn


class SkeletonExpert(nn.Module):
    """Coarse block-outline expert on top of the shared representation.

    Args:
        hidden_size: dimension of the draft backbone hidden states.
        skeleton_dim: rank of the sketch bottleneck.
        vocab_size: vocabulary size for the coarse outline distribution.
    """

    def __init__(self, *, hidden_size: int, skeleton_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.skeleton_dim = int(skeleton_dim)
        self.vocab_size = int(vocab_size)
        self.norm = nn.RMSNorm(self.hidden_size, eps=1e-6)
        self.proj = nn.Linear(self.hidden_size, self.skeleton_dim)
        self.act = nn.GELU()
        # Sketch vector returned to hidden space for the small experts.
        self.sketch_out = nn.Linear(self.skeleton_dim, self.hidden_size, bias=False)
        # Low-rank coarse outline distribution ("轮廓").
        self.coarse_head = nn.Linear(self.skeleton_dim, self.vocab_size, bias=False)

    def forward(self, hidden_states):
        """Args:
            hidden_states: [..., hidden_size] backbone hidden states.

        Returns:
            sketch: [..., hidden_size] sketch vectors conditioning the experts.
            coarse_logits: [..., vocab_size] coarse outline logits.
        """
        z = self.act(self.proj(self.norm(hidden_states)))
        return self.sketch_out(z), self.coarse_head(z)


__all__ = [
    "SkeletonExpert",
]
