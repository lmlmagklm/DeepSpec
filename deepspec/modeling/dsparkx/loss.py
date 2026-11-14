"""DSparkX training loss.

Composed from five weighted terms (all position-weighted by the same
``loss_weight_mask`` as DSpark, with optional exponential decay):

- ce_loss            -- CE of the confidence-merged proposal distribution.
- expert_ce_loss     -- mean CE over the K small experts (diversified监督).
- skeleton_ce_loss   -- CE of the general expert's coarse outline.
- confidence_loss    -- per-expert BCE against the per-expert accept rate.
- l1_loss            -- distribution matching between merged probs and target.
- recovery_ce_loss   -- clue-conditioned recovery CE ("不淘汰·补前面结构"):
                        given the simulated rejected candidate as the clue,
                        the re-completed prefix should predict the true token.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.utils.metrics import add_metric
from .common import DSparkXForwardOutput


_CONFIDENCE_KEYS = (
    "ce_loss_den",
    "expert_ce_den",
    "skeleton_ce_den",
    "l1_loss_den",
    "confidence_loss_den",
    "recovery_ce_den",
)


def _all_reduce_denominators(
    terms: dict[str, torch.Tensor],
    *,
    world_size: int,
) -> dict[str, torch.Tensor]:
    denominators = {}
    for key in _CONFIDENCE_KEYS:
        tensor = terms[key].detach().clone()
        if world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        denominators[key] = tensor
    return denominators


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    loss_decay_gamma: Optional[float],
) -> torch.Tensor:
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def _weighted_ce(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_logits = logits.reshape(-1, logits.size(-1))
    loss_per_token = F.cross_entropy(
        flat_logits.float(), target_ids.reshape(-1).long(), reduction="none"
    )
    weights = weight_mask.reshape(-1)
    return (loss_per_token * weights).sum(), weights.sum()


def compute_dsparkx_loss(
    *,
    outputs: DSparkXForwardOutput,
    loss_decay_gamma: Optional[float],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    expert_ce_alpha: float = 0.3,
    skeleton_ce_alpha: float = 0.1,
    clue_recovery_alpha: float = 0.3,
):
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    block_keep_mask = outputs.block_keep_mask
    num_experts = outputs.expert_logits.size(0)
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device
    zero = draft_logits.new_zeros((), dtype=torch.float32)

    weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
    )

    # Merged proposal CE.
    ce_num, ce_den = _weighted_ce(draft_logits, target_ids, weight_mask)

    # Per-expert CE (each expert supervised independently).
    expert_ce_num = zero.clone()
    expert_ce_den = zero.clone()
    if expert_ce_alpha > 0:
        for expert_idx in range(num_experts):
            num_i, den_i = _weighted_ce(
                outputs.expert_logits[expert_idx], target_ids, weight_mask
            )
            expert_ce_num = expert_ce_num + num_i
            expert_ce_den = expert_ce_den + den_i
        expert_ce_num = expert_ce_num / num_experts
        expert_ce_den = expert_ce_den / num_experts

    # Skeleton outline CE.
    skeleton_ce_num = zero.clone()
    skeleton_ce_den = zero.clone()
    if skeleton_ce_alpha > 0 and outputs.skeleton_logits is not None:
        skeleton_ce_num, skeleton_ce_den = _weighted_ce(
            outputs.skeleton_logits, target_ids, weight_mask
        )

    # Recovery CE (clue-conditioned prefix completion).
    recovery_ce_num = zero.clone()
    recovery_ce_den = zero.clone()
    if clue_recovery_alpha > 0 and outputs.recovery_logits is not None:
        recovery_ce_num, recovery_ce_den = _weighted_ce(
            outputs.recovery_logits, target_ids, weight_mask
        )

    # L1 distribution matching on the merged distribution.
    l1_num = zero.clone()
    l1_den = zero.clone()
    aligned_target_logits = outputs.aligned_target_logits
    assert l1_loss_alpha <= 0 or aligned_target_logits is not None, (
        "aligned_target_logits is required when l1_loss_alpha > 0."
    )
    accept_rate_3d = None
    if l1_loss_alpha > 0:
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
        l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
        l1_num = (l1_dist * weight_mask).sum()
        l1_den = weight_mask.sum()

    # Per-expert confidence BCE against per-expert accept rate.
    confidence_num = zero.clone()
    confidence_den = zero.clone()
    if confidence_head_alpha > 0 and outputs.expert_confidence is not None:
        assert aligned_target_logits is not None, (
            "aligned_target_logits is required when the confidence heads "
            "are enabled."
        )
        target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
        for expert_idx in range(num_experts):
            expert_probs = torch.softmax(
                outputs.expert_logits[expert_idx].float(), dim=-1
            )
            accept_rate = 1.0 - 0.5 * (expert_probs - target_probs).abs().sum(dim=-1)
            accept_rate = accept_rate.clamp(0.0, 1.0).detach()
            bce = F.binary_cross_entropy_with_logits(
                outputs.expert_confidence[expert_idx].float(),
                accept_rate,
                reduction="none",
            )
            confidence_num = confidence_num + (bce * weight_mask).sum()
            confidence_den = confidence_den + weight_mask.sum()
            with torch.no_grad():
                add_metric(
                    f"expert_conf_accept@{expert_idx}",
                    (accept_rate * weight_mask).sum(),
                    den=weight_mask.sum(),
                    tag="train",
                )
        confidence_num = confidence_num / num_experts
        confidence_den = confidence_den / num_experts

        with torch.no_grad():
            # Accept rate of the merged distribution for diagnostics.
            draft_probs = torch.softmax(draft_logits.float(), dim=-1)
            merged_accept = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
            merged_accept = merged_accept.clamp(0.0, 1.0)
            pos_accept = (merged_accept * eval_mask.to(torch.float32)).sum(dim=(0, 1))
            pos_total = eval_mask.to(torch.float32).sum(dim=(0, 1))
            for pos_idx in range(block_size):
                add_metric(
                    f"accept_rate@{pos_idx}",
                    pos_accept[pos_idx],
                    den=pos_total[pos_idx],
                    tag="train",
                )
            # Recovery accuracy lift: how often the clue-conditioned pass
            # fixes what the normal pass got wrong.
            if outputs.recovery_logits is not None:
                normal_hit = (
                    draft_logits.argmax(dim=-1) == target_ids
                ) & eval_mask
                recovery_hit = (
                    outputs.recovery_logits.argmax(dim=-1) == target_ids
                ) & eval_mask
                fixed = (recovery_hit & ~normal_hit).to(torch.float32).sum()
                broken = (normal_hit & ~recovery_hit).to(torch.float32).sum()
                add_metric("clue_fix", fixed, tag="train")
                add_metric("clue_break", broken, tag="train")

    terms = {
        "ce_loss_num": ce_num,
        "ce_loss_den": ce_den,
        "expert_ce_num": expert_ce_num,
        "expert_ce_den": expert_ce_den,
        "skeleton_ce_num": skeleton_ce_num,
        "skeleton_ce_den": skeleton_ce_den,
        "l1_loss_num": l1_num,
        "l1_loss_den": l1_den,
        "confidence_loss_num": confidence_num,
        "confidence_loss_den": confidence_den,
        "recovery_ce_num": recovery_ce_num,
        "recovery_ce_den": recovery_ce_den,
    }
    world_size = dist.get_world_size()
    global_den = _all_reduce_denominators(terms, world_size=world_size)

    def _scaled(num_key: str, den_key: str) -> torch.Tensor:
        den = global_den[den_key]
        if den.item() <= 0:
            return zero
        return terms[num_key] / (den + 1e-6)

    ce_loss = _scaled("ce_loss_num", "ce_loss_den")
    expert_ce_loss = _scaled("expert_ce_num", "expert_ce_den")
    skeleton_ce_loss = _scaled("skeleton_ce_num", "skeleton_ce_den")
    l1_loss = _scaled("l1_loss_num", "l1_loss_den")
    confidence_loss = _scaled("confidence_loss_num", "confidence_loss_den")
    recovery_ce_loss = _scaled("recovery_ce_num", "recovery_ce_den")

    local_loss = (
        float(ce_loss_alpha) * ce_loss
        + float(expert_ce_alpha) * expert_ce_loss
        + float(skeleton_ce_alpha) * skeleton_ce_loss
        + float(l1_loss_alpha) * l1_loss
        + float(confidence_head_alpha) * confidence_loss
        + float(clue_recovery_alpha) * recovery_ce_loss
    )
    # Match DSpark's DDP gradient scaling convention.
    backward_loss = local_loss * world_size

    add_metric("ce_loss", terms["ce_loss_num"], den=terms["ce_loss_den"], tag="train")
    add_metric(
        "expert_ce_loss",
        terms["expert_ce_num"],
        den=terms["expert_ce_den"],
        tag="train",
    )
    add_metric(
        "skeleton_ce_loss",
        terms["skeleton_ce_num"],
        den=terms["skeleton_ce_den"],
        tag="train",
    )
    if global_den["l1_loss_den"].item() > 0:
        add_metric("l1_loss", terms["l1_loss_num"], den=terms["l1_loss_den"], tag="train")
    if confidence_head_alpha > 0:
        add_metric(
            "confidence_loss",
            terms["confidence_loss_num"],
            den=terms["confidence_loss_den"],
            tag="train",
        )
    if clue_recovery_alpha > 0:
        add_metric(
            "recovery_ce_loss",
            terms["recovery_ce_num"],
            den=terms["recovery_ce_den"],
            tag="train",
        )
    add_metric("loss", local_loss, reduction="mean", tag="train")
    return backward_loss


__all__ = [
    "compute_dsparkx_loss",
]
