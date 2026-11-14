from __future__ import annotations

from types import SimpleNamespace

import torch

from deepspec.eval.base_evaluator import generate_decoding_sample
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.eval.dsparkx.draft_ops import (
    DSparkXDraftProposal,
    build_dsparkx_proposal,
    extract_rejection_clues,
    forward_dsparkx_draft_block,
)
from deepspec.modeling.dsparkx.qwen3 import Qwen3DSparkXModel


class Qwen3DSparkXEvaluator(Qwen3DSparkEvaluator):
    """DSparkX evaluator: skeleton -> K experts -> merged proposal with
    confidence truncation, plus the no-discard clue-repair loop.

    逐位裁决 (per-position target verification) is inherited unchanged from
    the DeepSpec speculative-decoding loop; DSparkX only changes what gets
    proposed and how rejections feed back into the next proposal.
    """

    draft_model_cls = Qwen3DSparkXModel

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.block_size)

    def _init_context(self, *, initial_output, **kwargs) -> SimpleNamespace:
        context = super()._init_context(initial_output=initial_output, **kwargs)
        # clue_token_ids: rejected candidates kept for the next proposal
        # ("候选留作线索，补齐前缀").
        context.clue_token_ids = None
        self._last_proposal: DSparkXDraftProposal | None = None
        return context

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DSparkXDraftProposal:
        model = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), self.max_proposal_tokens),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]

        clue_tensor = None
        if getattr(context, "clue_token_ids", None):
            clue_ids = context.clue_token_ids[-int(model.max_clue_tokens) :]
            clue_tensor = torch.tensor(
                [clue_ids], dtype=torch.long, device=output_ids.device
            )
            # 补齐前缀：place the clue tokens into the block prefix so the
            # experts decode the remaining positions on top of the structure.
            num_clue = min(clue_tensor.size(1), self.max_proposal_tokens - 1)
            draft_input_ids[:, 1 : 1 + num_clue] = clue_tensor[:, :num_clue]

        block_hidden = forward_dsparkx_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=self.max_proposal_tokens,
        )
        proposal = build_dsparkx_proposal(
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
            clue_token_ids=clue_tensor,
        )
        self._last_proposal = proposal
        return proposal

    def _update(self, context: SimpleNamespace, verification) -> None:
        super()._update(context, verification)
        # 不淘汰 · 补前面结构：rejected candidates become clues; a fully
        # accepted proposal clears the clue bank.
        if self._last_proposal is None:
            return
        context.clue_token_ids = extract_rejection_clues(
            self._last_proposal,
            verification,
            max_clue_tokens=int(self.draft_model.max_clue_tokens),
        )

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
        )


__all__ = [
    "Qwen3DSparkXEvaluator",
]
