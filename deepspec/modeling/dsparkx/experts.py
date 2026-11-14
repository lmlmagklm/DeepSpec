"""小专家池（Small Expert Pool）.

对应设计图中的「小专家 × K 并行解码，各出候选 + 置信度」：

- 每个小专家是一个低秩残差分支：输入 [H; sketch; clue?] 经过窄 hidden
  的 MLP 产出隐空间残差 delta，用门控 sigmoid 尺度化后加回共享隐状态，
  再共享 lm_head 得到候选分布。参数开销 = K × (d+s+c) × h 量级，
  不会复制 K 份全词表投影，保持「小专家」。
- 每个专家自带一个置信度头（AcceptRatePredictor），输出该位置候选的
  预期接受率 logit，供融合与逐位裁决前的动态截断使用。
- clue（线索）条件输入只在「不淘汰回补」路径使用：把被拒绝的候选 token
  的嵌入传给各专家，训练补齐前缀结构的能力。
"""

from typing import Optional

import torch
from torch import nn

from deepspec.modeling.dspark.common import AcceptRatePredictor


class SmallExpert(nn.Module):
    """A single low-rank residual draft expert with its own confidence head."""

    def __init__(
        self,
        *,
        hidden_size: int,
        skeleton_dim: int,
        expert_hidden_dim: int,
        use_clue: bool = True,
        confidence_with_markov_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_clue = bool(use_clue)
        in_dim = self.hidden_size + int(skeleton_dim)
        if self.use_clue:
            in_dim += self.hidden_size
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(expert_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(expert_hidden_dim), self.hidden_size),
        )
        self.gate_proj = nn.Linear(in_dim, 1)
        conf_input_dim = self.hidden_size + int(confidence_with_markov_rank)
        self.confidence_head = AcceptRatePredictor(input_dim=conf_input_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sketch: torch.Tensor,
        clue_embed: Optional[torch.Tensor] = None,
        markov_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            hidden_states: [..., hidden_size] backbone hidden states (H).
            sketch: [..., hidden_size] sketch vectors from the skeleton expert.
            clue_embed: optional [..., hidden_size] clue token embeddings; when
                provided this is a recovery ("补前面结构") forward pass.
            markov_features: optional [..., markov_rank] prev-token embeddings
                concatenated onto the confidence-head input.

        Returns:
            expert_hidden: [..., hidden_size] residual-adjusted hidden states,
                to be projected by the shared lm_head for candidate logits.
            confidence: [...,] confidence logits (pre-sigmoid).
        """
        parts = [hidden_states, sketch]
        if self.use_clue:
            if clue_embed is None:
                clue_embed = torch.zeros_like(hidden_states)
            parts.append(clue_embed.to(dtype=hidden_states.dtype))
        gate_input = torch.cat(parts, dim=-1)
        delta = self.mlp(gate_input)
        gate = torch.sigmoid(self.gate_proj(gate_input)).to(delta.dtype)
        expert_hidden = hidden_states + gate * delta

        conf_features = expert_hidden
        if markov_features is not None:
            conf_features = torch.cat(
                [expert_hidden, markov_features.to(expert_hidden.dtype)],
                dim=-1,
            )
        confidence = self.confidence_head(conf_features).float()
        return expert_hidden, confidence


class SmallExpertPool(nn.Module):
    """K small experts decoded in parallel over the shared representation."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        skeleton_dim: int,
        expert_hidden_dim: int,
        use_clue: bool = True,
        confidence_with_markov_rank: int = 0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        assert self.num_experts > 0, "num_experts must be positive."
        self.experts = nn.ModuleList(
            [
                SmallExpert(
                    hidden_size=hidden_size,
                    skeleton_dim=skeleton_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    use_clue=use_clue,
                    confidence_with_markov_rank=confidence_with_markov_rank,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sketch: torch.Tensor,
        clue_embed: Optional[torch.Tensor] = None,
        markov_features: Optional[torch.Tensor] = None,
        lm_head: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all K experts in parallel.

        Args:
            hidden_states: [..., hidden_size] backbone hidden states.
            sketch: [..., hidden_size] skeleton sketch vectors.
            clue_embed: optional clue token embeddings for recovery passes.
            markov_features: optional prev-token features for confidence heads.
            lm_head: shared lm_head used to project each expert's hidden state
                into candidate logits. When None, logits are not computed and
                the first return value holds the stacked expert hidden states.

        Returns:
            stacked: [num_experts, ..., vocab_size] candidate logits when
                lm_head is given, else [num_experts, ..., hidden_size].
            confidence: [num_experts, ...] per-expert confidence logits.
        """
        expert_outputs = []
        confidence_outputs = []
        for expert in self.experts:
            expert_hidden, confidence = expert(
                hidden_states,
                sketch,
                clue_embed=clue_embed,
                markov_features=markov_features,
            )
            if lm_head is not None:
                expert_outputs.append(lm_head(expert_hidden))
            else:
                expert_outputs.append(expert_hidden)
            confidence_outputs.append(confidence)
        return torch.stack(expert_outputs, dim=0), torch.stack(
            confidence_outputs, dim=0
        )


__all__ = [
    "SmallExpert",
    "SmallExpertPool",
]
