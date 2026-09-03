import torch
from .objective import Objective
import torch.nn.functional as F

class CrossEntropyObjective(Objective):
    name = "cross_entropy"

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        loss = F.cross_entropy(logits, y_true)
        return [loss], [self.name]

    
class KLPreservationObjective(Objective):
    name = "kl_preservation"

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs,
        input_type,
    ):
        if input_type == "probability":
            base_probs = base_outputs
        elif input_type == "score":
            base_probs = F.softmax(base_outputs, dim=1)
        else:
            raise ValueError(
                "input_type deve ser 'probability' ou 'score'."
            )

        corrected_log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        loss = F.kl_div(
            corrected_log_probs,
            base_probs,
            reduction="batchmean",
        )

        return [loss], [self.name]

class JensenShannonPreservationObjective(Objective):
    name = "jensen_shannon_preservation"

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        if base_outputs is None:
            raise ValueError(
                "JensenShannonPreservationObjective requer "
                "base_outputs."
            )

        if input_type == "probability":
            base_probs = base_outputs

        elif input_type == "score":
            base_probs = F.softmax(
                base_outputs,
                dim=1,
            )

        else:
            raise ValueError(
                "input_type deve ser 'probability' ou 'score'."
            )

        corrected_probs = F.softmax(
            logits,
            dim=1,
        )

        mixture_probs = 0.5 * (
            base_probs + corrected_probs
        )

        base_log_probs = torch.log(
            base_probs.clamp_min(1e-8)
        )

        corrected_log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        mixture_log_probs = torch.log(
            mixture_probs.clamp_min(1e-8)
        )

        kl_base_mixture = torch.sum(
            base_probs
            * (
                base_log_probs
                - mixture_log_probs
            ),
            dim=1,
        ).mean()

        kl_corrected_mixture = torch.sum(
            corrected_probs
            * (
                corrected_log_probs
                - mixture_log_probs
            ),
            dim=1,
        ).mean()

        loss = 0.5 * (
            kl_base_mixture
            + kl_corrected_mixture
        )

        return [loss], [self.name]