from deepspec.data import CacheCollator
from deepspec.modeling.dsparkx.loss import compute_dsparkx_loss
from deepspec.modeling.dsparkx.qwen3 import (
    Qwen3DSparkXModel,
    build_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkXTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkXModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dsparkx_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
            expert_ce_alpha=float(self.args.model.get("expert_ce_alpha", 0.3)),
            skeleton_ce_alpha=float(self.args.model.get("skeleton_ce_alpha", 0.1)),
            clue_recovery_alpha=float(
                self.args.model.get("clue_recovery_alpha", 0.3)
            ),
        )
        return loss


__all__ = [
    "Qwen3DSparkXTrainer",
]
