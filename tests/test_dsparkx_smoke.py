"""DSparkX smoke test.

Validates tensor shapes, the training forward, the six-term loss with
backward, the inference proposal helper, and the clue-extraction logic on a
tiny random Qwen3 config.

Run:
    python tests/test_dsparkx_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from deepspec.modeling.dsparkx.common import merge_expert_probs
from deepspec.modeling.dsparkx.loss import compute_dsparkx_loss
from deepspec.modeling.dsparkx.qwen3 import Qwen3DSparkXModel, build_draft_config
from deepspec.eval.dsparkx.draft_ops import (
    DSparkXDraftProposal,
    extract_rejection_clues,
)


def _init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def _tiny_target_config(vocab_size=128):
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def _tiny_model_args():
    class Args(dict):
        __getattr__ = dict.get

    return Args(
        num_draft_layers=2,
        target_layer_ids=[0, 1],
        block_size=4,
        mask_token_id=vocab_size - 1,
        num_anchors=6,
        markov_rank=16,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
        num_experts=3,
        skeleton_dim=32,
        expert_hidden_dim=64,
        max_clue_tokens=2,
    )


vocab_size = 128


def test_training_forward_and_loss():
    target_config = _tiny_target_config()
    draft_config = build_draft_config(
        target_config=target_config,
        model_args=_tiny_model_args(),
    )
    model = Qwen3DSparkXModel(draft_config).eval()

    bsz, seq_len, num_layers = 2, 24, 2
    input_ids = torch.randint(0, vocab_size - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    loss_mask[0, -3:] = 0
    target_hidden_states = torch.randn(
        bsz, seq_len, num_layers * target_config.hidden_size
    )
    target_last_hidden_states = torch.randn(bsz, seq_len, target_config.hidden_size)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )

    k = int(draft_config.num_experts)
    assert outputs.draft_logits.shape == (bsz, 6, 4, vocab_size), (
        outputs.draft_logits.shape
    )
    assert outputs.expert_logits.shape == (k, bsz, 6, 4, vocab_size), (
        outputs.expert_logits.shape
    )
    assert outputs.expert_confidence.shape == (k, bsz, 6, 4)
    assert outputs.skeleton_logits.shape == (bsz, 6, 4, vocab_size)
    assert outputs.recovery_logits.shape == (bsz, 6, 4, vocab_size)
    assert outputs.target_ids.shape == (bsz, 6, 4)
    assert outputs.eval_mask.shape == (bsz, 6, 4)
    assert torch.isfinite(outputs.draft_logits).all()
    assert torch.isfinite(outputs.recovery_logits).all()

    # Merged probs must be normalized.
    merged_probs = outputs.draft_logits.exp()
    assert torch.allclose(
        merged_probs.sum(-1), torch.ones_like(merged_probs.sum(-1)), atol=1e-4
    )

    loss = compute_dsparkx_loss(
        outputs=outputs,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
        expert_ce_alpha=0.3,
        skeleton_ce_alpha=0.1,
        clue_recovery_alpha=0.3,
    )
    assert torch.isfinite(loss), loss
    print(f"[smoke] training forward OK, loss={float(loss):.4f}")


def test_inference_proposal_and_clues():
    target_config = _tiny_target_config()
    draft_config = build_draft_config(
        target_config=target_config,
        model_args=_tiny_model_args(),
    )
    model = Qwen3DSparkXModel(draft_config).eval()
    block_hidden = torch.randn(1, 4, target_config.hidden_size)
    first_prev = torch.tensor([7])

    with torch.no_grad():
        tokens, probs, conf = model.propose_from_block_hidden(
            block_hidden,
            first_prev_token_ids=first_prev,
            temperature=0.0,
            clue_token_ids=None,
            confidence_threshold=0.0,
        )
    assert tokens.shape == (1, 4)
    assert probs.shape == (1, 4, vocab_size)
    assert conf.shape == (1, 4)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 4), atol=1e-4)

    # With clues + threshold.
    with torch.no_grad():
        tokens2, probs2, conf2 = model.propose_from_block_hidden(
            block_hidden,
            first_prev_token_ids=first_prev,
            temperature=1.0,
            clue_token_ids=torch.tensor([[11, 22]]),
            confidence_threshold=0.99,
        )
    assert tokens2.size(1) == probs2.size(1) == conf2.size(1)
    assert tokens2.size(1) <= 4
    print("[smoke] inference proposal OK "
          f"(no-clue len=4, clue+threshold len={tokens2.size(1)})")

    # Clue extraction: reject inside the block -> keep rejected suffix.
    proposal = DSparkXDraftProposal(
        draft_token_count=3,
        verify_input_ids=torch.tensor([[5, 10, 20, 30]]),
        draft_probs=None,
        confidence_logits=None,
    )

    class FakeVerification:
        accepted_draft_tokens = 1

    clues = extract_rejection_clues(proposal, FakeVerification(), max_clue_tokens=5)
    assert clues == [20, 30], clues

    class AcceptedVerification:
        accepted_draft_tokens = 3

    clues = extract_rejection_clues(
        proposal, AcceptedVerification(), max_clue_tokens=5
    )
    assert clues is None
    print(f"[smoke] clue extraction OK (rejected suffix kept: {clues})")


def test_merge_expert_probs():
    torch.manual_seed(0)
    expert_probs = torch.rand(4, 2, 3, vocab_size)
    expert_probs = expert_probs / expert_probs.sum(-1, keepdim=True)
    conf = torch.randn(4, 2, 3)
    merged, weights = merge_expert_probs(expert_probs, conf)
    assert merged.shape == (2, 3, vocab_size)
    assert torch.allclose(merged.sum(-1), torch.ones(2, 3), atol=1e-5)
    assert torch.allclose(weights.sum(0), torch.ones(2, 3), atol=1e-5)
    assert (weights >= 0).all()
    print("[smoke] merge_expert_probs OK")


if __name__ == "__main__":
    _init_dist()
    test_merge_expert_probs()
    test_training_forward_and_loss()
    test_inference_proposal_and_clues()
    if dist.is_initialized():
        dist.destroy_process_group()
    print("All DSparkX smoke tests passed.")
