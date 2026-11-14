"""DSparkX decode-time operators.

Implements the right half of the DSparkX design diagram:

- K small experts decode in parallel from the shared representation +
  skeleton sketch, each emitting a candidate distribution and a confidence.
- The merged distribution is sampled autoregressive-position by position
  (Markov bias applied per step), with confidence-threshold truncation
  identical in spirit to DSpark.
- ``不淘汰 · 补前面结构``: rejected candidates are kept as clues. The next
  proposal places the clue tokens into the block prefix so the experts decode
  the remaining positions on top of the completed structure ("再上").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepspec.eval.base_evaluator import DraftProposal
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
from deepspec.modeling.dsparkx.qwen3.modeling import Qwen3DSparkXModel


@dataclass
class DSparkXDraftProposal(DraftProposal):
    # Merged confidence logits per proposed position.
    confidence_logits: torch.Tensor | None = None
    # Number of leading block positions occupied by clue tokens (not verified).
    num_clue_tokens: int = 0


def forward_dsparkx_draft_block(
    model: Qwen3DSparkXModel,
    *,
    draft_input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values_draft,
    target_hidden_states: torch.Tensor,
    start: int,
    block_size: int,
):
    """Backbone forward for one draft block; reusable as-is from DSpark."""
    return forward_dspark_draft_block(
        model=model,
        draft_input_ids=draft_input_ids,
        position_ids=position_ids,
        past_key_values_draft=past_key_values_draft,
        target_hidden_states=target_hidden_states,
        start=start,
        block_size=block_size,
    )


def build_dsparkx_proposal(
    model: Qwen3DSparkXModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
    clue_token_ids: torch.Tensor | None = None,
) -> DSparkXDraftProposal:
    """Skeleton -> K experts -> merged candidates, with clue-conditioned prefix.

    ``draft_input_ids`` carries the clue tokens at positions
    ``1 : 1 + num_clue``; only the sampled candidates after the clue prefix
    are proposed for verification.
    """
    assert draft_input_ids.size(0) == 1, "build_dsparkx_proposal requires batch_size=1"
    num_clue = 0
    if clue_token_ids is not None and clue_token_ids.size(1) > 0:
        num_clue = min(int(clue_token_ids.size(1)), block_size - 1)

    proposal_hidden_states = block_hidden[:, :block_size, :]
    sampled_tokens, draft_probs, merged_conf = model.propose_from_block_hidden(
        proposal_hidden_states,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        clue_token_ids=clue_token_ids,
        confidence_threshold=confidence_threshold,
    )

    num_candidates = sampled_tokens.size(1)
    if num_candidates == 0:
        return DSparkXDraftProposal(
            draft_token_count=0,
            verify_input_ids=draft_input_ids[:, :1],
            draft_probs=None,
            confidence_logits=None,
            num_clue_tokens=num_clue,
        )

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens],
        dim=1,
    )
    return DSparkXDraftProposal(
        draft_token_count=num_candidates,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=merged_conf,
        num_clue_tokens=num_clue,
    )


def extract_rejection_clues(
    proposal: DSparkXDraftProposal,
    verification,
    *,
    max_clue_tokens: int,
) -> list[int] | None:
    """不淘汰 · 补前面结构：keep rejected candidates as clues.

    When the adjudication stops inside the proposed block, the rejected
    suffix is not discarded: it is returned as the clue for the next proposal
    round, so the experts can complete the prefix structure and re-propose.
    """
    accepted = int(verification.accepted_draft_tokens)
    proposed = int(proposal.draft_token_count)
    if proposed <= 0 or accepted >= proposed:
        return None
    rejected = proposal.verify_input_ids[0, accepted + 1 : proposed + 1]
    if max_clue_tokens > 0:
        rejected = rejected[-max_clue_tokens:]
    return [int(token_id) for token_id in rejected.tolist()]


__all__ = [
    "DSparkXDraftProposal",
    "build_dsparkx_proposal",
    "extract_rejection_clues",
    "forward_dsparkx_draft_block",
]
