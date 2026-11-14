"""Shared components for DSparkX.

DSparkX upgrades the DSpark draft model with a three-stage decoding pipeline
(see ``docs/dsparkx_design.md``):

1. 共享向量表征 H   -- inherited from DSpark: multi-layer target hidden states
                      fused by ``fc`` + ``hidden_norm``.
2. 通用专家出轮廓   -- ``SkeletonExpert`` produces a coarse block-level sketch.
3. K 小专家并行     -- ``SmallExpertPool`` decodes K candidates in parallel,
                      each with its own confidence head; candidates are merged
                      by confidence-weighted mixing.
4. 逐位裁决         -- the target model verifies tokens position by position
                      (unchanged rejection sampling from DeepSpec).
5. 不淘汰回补       -- rejected candidates are kept as clues (``clue``) that
                      condition the next proposal round to complete the prefix
                      structure before re-proposing ("再上").
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DSparkXForwardOutput:
    """Outputs for one DSparkX training forward.

    Shape symbols:
        batch_size: number of samples in the batch
        num_anchors: sampled anchor blocks per sample
        block_size: number of draft positions per anchor
        num_experts: number of parallel small experts (K)
        vocab_size: vocabulary size
    """

    # Merged draft distribution as log-probs.
    # [batch_size, num_anchors, block_size, vocab_size]
    draft_logits: torch.Tensor
    # Per-expert logits before merging, stacked on dim 0.
    # [num_experts, batch_size, num_anchors, block_size, vocab_size]
    expert_logits: torch.Tensor
    # Per-expert confidence logits (pre-sigmoid accept-rate prediction).
    # [num_experts, batch_size, num_anchors, block_size]
    expert_confidence: Optional[torch.Tensor] = None
    # Coarse skeleton ("outline") logits from the general expert.
    # [batch_size, num_anchors, block_size, vocab_size]
    skeleton_logits: Optional[torch.Tensor] = None
    # Clue-conditioned recovery logits (log-probs) used to train the
    # "不淘汰·补前面结构" behavior: given the rejected candidate as a clue,
    # re-complete the prefix structure and predict the true tokens.
    # [batch_size, num_anchors, block_size, vocab_size]
    recovery_logits: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size]
    target_ids: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size]
    eval_mask: Optional[torch.Tensor] = None
    # [batch_size, num_anchors]
    block_keep_mask: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size, vocab_size]
    aligned_target_logits: Optional[torch.Tensor] = None


def confidence_mixing_weights(
    expert_confidence: torch.Tensor,
) -> torch.Tensor:
    """Normalize per-expert confidences into mixing weights.

    Args:
        expert_confidence: [K, ...] confidence logits (pre-sigmoid).

    Returns:
        [K, ...] non-negative weights summing to 1 over dim 0.
    """
    weights = torch.sigmoid(expert_confidence.float())
    return weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)


def merge_expert_probs(
    expert_probs: torch.Tensor,
    expert_confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-expert token distributions by confidence-weighted mixing.

    Args:
        expert_probs: [K, ..., vocab_size] per-expert probability
            distributions (each already normalized over the last dim).
        expert_confidence: [K, ...] confidence logits (pre-sigmoid).

    Returns:
        merged_probs: [..., vocab_size] normalized mixture distribution.
        mixing_weights: [K, ...] normalized expert weights (for logging).
    """
    weights = confidence_mixing_weights(expert_confidence)
    merged = (weights.unsqueeze(-1) * expert_probs.float()).sum(dim=0)
    merged = merged / merged.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return merged, weights


def merged_confidence(
    expert_confidence: torch.Tensor,
) -> torch.Tensor:
    """Confidence of the merged distribution (mixture of per-expert sigmoids).

    Args:
        expert_confidence: [K, ...] confidence logits.

    Returns:
        [...] scalar confidence logit per position.
    """
    weights = confidence_mixing_weights(expert_confidence)
    probs = torch.sigmoid(expert_confidence.float())
    return (weights * probs).sum(dim=0)


__all__ = [
    "DSparkXForwardOutput",
    "confidence_mixing_weights",
    "merge_expert_probs",
    "merged_confidence",
]
