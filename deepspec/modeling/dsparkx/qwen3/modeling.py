from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache

from deepspec.modeling.dspark.common import (
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.modeling.dsparkx.common import (
    DSparkXForwardOutput,
    merge_expert_probs,
    merged_confidence,
)
from deepspec.modeling.dsparkx.experts import SmallExpertPool
from deepspec.modeling.dsparkx.skeleton import SkeletonExpert
from deepspec.utils.sampling import sample_from_probs


class Qwen3DSparkXModel(Qwen3DSparkModel):
    """DSparkX draft model: skeleton general expert + K parallel small experts
    + confidence-weighted merging + clue-conditioned recovery.

    Training-time ``forward`` mirrors the DSpark forward (shared representation
    H -> backbone over mask blocks) and then runs the DSparkX head stack:

        skeleton sketch -> K experts (parallel) -> merged proposal
        -> simulated-rejection clue -> clue-conditioned recovery pass

    The parent's single confidence head is replaced by per-expert confidence
    heads owned by ``self.expert_pool``.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_experts = int(config.num_experts)
        assert self.num_experts > 0, "config.num_experts must be positive."

        # Replace the parent's single confidence head with per-expert ones.
        self.confidence_head = None
        self.enable_confidence_head = False
        self.experts_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )
        confidence_markov_rank = (
            int(config.markov_rank) if self.experts_with_markov else 0
        )

        # 通用专家 · 出轮廓.
        self.skeleton = SkeletonExpert(
            hidden_size=config.hidden_size,
            skeleton_dim=int(config.skeleton_dim),
            vocab_size=config.vocab_size,
        )
        # 小专家 × K 并行解码.
        self.expert_pool = SmallExpertPool(
            num_experts=self.num_experts,
            hidden_size=config.hidden_size,
            skeleton_dim=int(config.skeleton_dim),
            expert_hidden_dim=int(config.expert_hidden_dim),
            use_clue=True,
            confidence_with_markov_rank=confidence_markov_rank,
        )
        self.clue_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.max_clue_tokens = int(getattr(config, "max_clue_tokens", 0))

    # ------------------------------------------------------------------
    # Shared helpers.
    # ------------------------------------------------------------------
    def _compute_markov_bias(
        self,
        ref_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Pure Markov step bias (shared across experts)."""
        if self.markov_head is None:
            return None
        zero_logits = torch.zeros_like(ref_logits)
        return self.markov_head.apply_block_logits(
            zero_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def _expert_markov_features(
        self,
        prev_token_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.experts_with_markov or self.markov_head is None:
            return None
        return self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype)

    # ------------------------------------------------------------------
    # Training forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkXForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(
            seq_len, device=device
        ).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(-1, num_blocks, -1, -1),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, target_last_hidden_states.size(-1)
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        # 通用专家 · 出轮廓.
        sketch, skeleton_logits = self.skeleton(output_hidden_4d)

        # K 小专家并行解码（第一轮：无线索的正常提案）.
        markov_features = self._expert_markov_features(
            prev_token_ids, output_hidden_4d.dtype
        )
        expert_logits, expert_confidence = self.expert_pool(
            output_hidden_4d,
            sketch,
            markov_features=markov_features,
            lm_head=self.lm_head,
        )
        markov_bias = self._compute_markov_bias(
            expert_logits[0],
            prev_token_ids=prev_token_ids,
            hidden_states=output_hidden_4d,
        )
        if markov_bias is not None:
            expert_logits = expert_logits + markov_bias.unsqueeze(0)

        expert_logits = expert_logits.to(torch.bfloat16)
        expert_probs = torch.softmax(expert_logits.float(), dim=-1)
        merged_probs, _ = merge_expert_probs(expert_probs, expert_confidence)
        draft_logits = torch.log(merged_probs.clamp_min(1e-9))

        # 不淘汰 · 补前面结构：模拟一次逐位裁决，把合并分布的 argmax 当作
        # "被拒绝的候选"线索，训练线索条件下的前缀回补能力.
        with torch.no_grad():
            clue_ids = merged_probs.argmax(dim=-1)
        clue_embed = self.clue_norm(self.embed_tokens(clue_ids))
        recovery_expert_logits, recovery_confidence = self.expert_pool(
            output_hidden_4d,
            sketch,
            clue_embed=clue_embed,
            markov_features=markov_features,
            lm_head=self.lm_head,
        )
        if markov_bias is not None:
            recovery_expert_logits = recovery_expert_logits + markov_bias.unsqueeze(0)
        recovery_expert_logits = recovery_expert_logits.to(torch.bfloat16)
        recovery_probs = torch.softmax(recovery_expert_logits.float(), dim=-1)
        recovery_merged, _ = merge_expert_probs(recovery_probs, recovery_confidence)
        recovery_logits = torch.log(recovery_merged.clamp_min(1e-9))

        return DSparkXForwardOutput(
            draft_logits=draft_logits,
            expert_logits=expert_logits,
            expert_confidence=expert_confidence,
            skeleton_logits=skeleton_logits.to(torch.bfloat16),
            recovery_logits=recovery_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            aligned_target_logits=aligned_target_logits,
        )

    # ------------------------------------------------------------------
    # Inference helpers (used by the DSparkX evaluator).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def propose_from_block_hidden(
        self,
        block_hidden: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float,
        clue_token_ids: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Skeleton -> K experts -> merged autoregressive sampling.

        Args:
            block_hidden: [batch, block_size, hidden_size] backbone outputs.
            first_prev_token_ids: [batch] token preceding the first draft
                position (used by the Markov head).
            temperature: sampling temperature.
            clue_token_ids: optional [batch, num_clue] rejected candidates kept
                as clues ("候选留作线索"). Position k of the block is
                conditioned on ``clue_token_ids[:, k]`` for k < num_clue.
            confidence_threshold: truncate the proposal at the first position
                whose merged confidence falls below this threshold.

        Returns:
            sampled_tokens: [batch, block_size] merged candidates.
            draft_probs: [batch, block_size, vocab_size] merged probabilities.
            merged_conf: [batch, block_size] merged confidence logits.
        """
        batch_size, proposal_len, hidden_size = block_hidden.shape
        device = block_hidden.device
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size, 0, dtype=torch.long, device=device
            )
            return (
                empty_tokens,
                torch.empty(batch_size, 0, self.config.vocab_size, device=device),
                torch.empty(batch_size, 0, device=device),
            )

        sketch, _ = self.skeleton(block_hidden)

        clue_embed = None
        if clue_token_ids is not None and clue_token_ids.size(1) > 0:
            num_clue = min(clue_token_ids.size(1), proposal_len)
            clue_embed = torch.zeros_like(block_hidden)
            clue_embed[:, :num_clue, :] = self.clue_norm(
                self.embed_tokens(clue_token_ids[:, :num_clue].long())
            ).to(block_hidden.dtype)

        expert_logits, expert_confidence = self.expert_pool(
            block_hidden,
            sketch,
            clue_embed=clue_embed,
            markov_features=None,
            lm_head=self.lm_head,
        )
        expert_logits = expert_logits.float()

        sampled_tokens = []
        draft_probs = []
        confidences = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            if self.markov_head is not None:
                step_bias = self.markov_head.compute_step_bias(
                    prev_token_ids,
                    block_hidden[:, step_idx, :],
                )
                step_logits = expert_logits[:, :, step_idx, :] + step_bias.unsqueeze(0)
            else:
                step_logits = expert_logits[:, :, step_idx, :]
            if temperature < 1e-5:
                step_probs = torch.zeros_like(step_logits)
                step_probs.scatter_(
                    -1, torch.argmax(step_logits, dim=-1, keepdim=True), 1.0
                )
            else:
                step_probs = torch.softmax(step_logits / temperature, dim=-1)
            merged_probs, _ = merge_expert_probs(
                step_probs, expert_confidence[:, :, step_idx]
            )
            conf_step = merged_confidence(expert_confidence[:, :, step_idx])
            sampled = sample_from_probs(merged_probs.unsqueeze(1)).squeeze(1)
            sampled_tokens.append(sampled)
            draft_probs.append(merged_probs)
            confidences.append(conf_step)
            prev_token_ids = sampled

        sampled_tokens = torch.stack(sampled_tokens, dim=1)
        draft_probs = torch.stack(draft_probs, dim=1)
        merged_conf = torch.stack(confidences, dim=1)

        if confidence_threshold > 0.0:
            below = merged_conf.sigmoid() < confidence_threshold
            if bool(below[0].any().item()):
                cut = int(torch.nonzero(below[0], as_tuple=False)[0].item())
                sampled_tokens = sampled_tokens[:, :cut]
                draft_probs = draft_probs[:, :cut]
                merged_conf = merged_conf[:, :cut]

        return sampled_tokens, draft_probs, merged_conf


__all__ = [
    "Qwen3DSparkXModel",
]
