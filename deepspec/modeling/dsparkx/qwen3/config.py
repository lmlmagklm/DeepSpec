import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    """Build the draft config for Qwen3DSparkXModel.

    Inherits the DSpark draft config and adds the DSparkX fields:
        num_experts              -- K parallel small experts.
        skeleton_dim             -- bottleneck rank of the skeleton expert.
        expert_hidden_dim        -- hidden width of each small expert MLP.
        max_clue_tokens          -- max rejected candidates kept as clues.
        expert_ce_alpha          -- per-expert CE weight (training loss).
        skeleton_ce_alpha        -- skeleton outline CE weight.
        clue_recovery_alpha      -- clue-conditioned recovery CE weight.
    """
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    num_experts = int(getattr(model_args, "num_experts", 4))
    assert num_experts > 0, f"num_experts must be positive, got {num_experts}."
    skeleton_dim = int(getattr(model_args, "skeleton_dim", 512))
    expert_hidden_dim = int(getattr(model_args, "expert_hidden_dim", 1024))
    max_clue_tokens = int(getattr(model_args, "max_clue_tokens", 0))
    block_size = int(model_args.block_size)
    if max_clue_tokens <= 0:
        # Keep at most half of the block as clue prefix by default.
        max_clue_tokens = max(block_size // 2, 1)

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3DSparkXModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = block_size
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    # DSparkX specific fields.
    draft_config.num_experts = num_experts
    draft_config.skeleton_dim = skeleton_dim
    draft_config.expert_hidden_dim = expert_hidden_dim
    draft_config.max_clue_tokens = max_clue_tokens
    return draft_config


__all__ = [
    "build_draft_config",
]
