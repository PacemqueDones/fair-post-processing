import torch
import torch.nn.functional as F

from .objective import Objective
from .reduction import reduce_losses
from .validation import (
    prepare_binary_sensitive_attributes,
)


class DemographicParityObjective(Objective):
    name = "demographic_parity"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0.0,
        relaxation="mean",
        sensitive_indices=None,
        reduction="none",
        eps=1e-7,
    ):
        valid_relaxations = {
            "mean",
            "kl",
        }

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be either "
                "'mean' or 'kl'."
            )

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.relaxation = relaxation
        self.sensitive_indices = sensitive_indices
        self.reduction = reduction
        self.eps = eps

        self.name = (
            f"demographic_parity_{relaxation}"
        )

    def _kl_bern(
        self,
        p0,
        p1,
    ):
        p0 = torch.clamp(
            p0,
            min=self.eps,
            max=1.0 - self.eps,
        )

        p1 = torch.clamp(
            p1,
            min=self.eps,
            max=1.0 - self.eps,
        )

        return (
            p0 * torch.log(p0 / p1)
            + (1.0 - p0)
            * torch.log(
                (1.0 - p0)
                / (1.0 - p1)
            )
        )

    def _compute_attribute_loss(
        self,
        preds_pos,
        attribute,
        logits,
    ):
        group0 = preds_pos[
            attribute == 0
        ]

        group1 = preds_pos[
            attribute == 1
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            return logits.sum() * 0.0

        p0 = group0.mean()
        p1 = group1.mean()

        if self.relaxation == "mean":
            return torch.abs(
                p0 - p1
            )

        return (
            self._kl_bern(p0, p1)
            + self._kl_bern(p1, p0)
        )

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
    ):
        (
            sensitive_attr,
            sensitive_indices,
        ) = prepare_binary_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=logits.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        preds_pos = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        if self.ce_weight > 0:
            ce = F.cross_entropy(
                logits,
                y_true,
            )
        else:
            ce = logits.sum() * 0.0

        losses = []
        names = []

        for local_index, original_index in enumerate(
            sensitive_indices
        ):
            attribute = sensitive_attr[
                :,
                local_index,
            ]

            fairness = self._compute_attribute_loss(
                preds_pos=preds_pos,
                attribute=attribute,
                logits=logits,
            )

            loss = (
                self.fairness_weight * fairness
                + self.ce_weight * ce
            )

            loss_name = (
                f"{self.name}_attribute_"
                f"{original_index}"
            )

            losses.append(loss)
            names.append(loss_name)

        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.reduction,
            reduced_name=self.name,
        )
    
class EqualityOpportunityObjective(Objective):
    name = "equality_opportunity"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0.0,
        relaxation="mean",
        eps=1e-7,
    ):
        valid_relaxations = {"mean", "kl"}

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be either  'mean' or 'kl'."
            )

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.relaxation = relaxation
        self.eps = eps

        self.name = f"equality_opportunity_{relaxation}"

    def _kl_bern(self, p0, p1):
        p0 = torch.clamp(p0, min=self.eps, max=1.0 - self.eps,)

        p1 = torch.clamp(p1, min=self.eps, max=1.0 - self.eps,)

        return (p0 * torch.log(p0 / p1) + (1.0 - p0) * torch.log((1.0 - p0) / (1.0 - p1)))

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
    ):
        
        self._validate_binary_sensitive_attr(
            sensitive_attr=sensitive_attr,
            num_samples=logits.shape[0],
        )

        preds_pos = torch.softmax(logits, dim=1, )[:, 1]

        mask_pos = y_true == 1

        group0 = preds_pos[(sensitive_attr == 0) & mask_pos]

        group1 = preds_pos[(sensitive_attr == 1) & mask_pos]

        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, y_true)
        else:
            ce = 0

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0

        else:
            p0 = group0.mean()
            p1 = group1.mean()

            if self.relaxation == "mean":
                fairness = torch.abs(p0 - p1)

            else:
                fairness = (self._kl_bern(p0, p1) + self._kl_bern(p1, p0))

        loss = (self.fairness_weight * fairness + self.ce_weight * ce)

        return [loss], [self.name] 
    
class WassersteinEqualityOpportunityObjective(Objective):
    name = "equality_opportunity_wasserstein"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=1,
        blur=0.05,
        scaling=0.9,
        debias=True,
        backend="auto",
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

        self.wasserstein = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )

    def __call__(self, logits, y_true, sensitive_attr):
        self._validate_binary_sensitive_attr(
            sensitive_attr=sensitive_attr,
            num_samples=logits.shape[0],
        )
                
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            group0 = group0.reshape(-1, 1)
            group1 = group1.reshape(-1, 1)

            fairness = self.wasserstein(group0, group1)

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
    
class WassersteinEqualityOpportunityObjective(Objective):
    name = "equality_opportunity_wasserstein"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=1,
        blur=0.05,
        scaling=0.9,
        debias=True,
        backend="auto",
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight

        self.wasserstein = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            group0 = group0.reshape(-1, 1)
            group1 = group1.reshape(-1, 1)

            fairness = self.wasserstein(group0, group1)

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )
    
class WassersteinEqualityOpportunityQuantileObjective(Objective):
    name = "equality_opportunity_wasserstein_quantile"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0,
        p=2,
        num_quantiles=10**4,
    ):
        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.p = p
        self.num_quantiles = num_quantiles

    def __call__(self, logits, y_true, sensitive_attr):
        preds_pos = torch.softmax(logits, dim=1)[:, 1]
        mask_pos = y_true == 1

        group0 = preds_pos[
            (sensitive_attr == 0) & mask_pos
        ]

        group1 = preds_pos[
            (sensitive_attr == 1) & mask_pos
        ]

        if group0.numel() == 0 or group1.numel() == 0:
            fairness = logits.sum() * 0.0
        else:
            quantile_levels = torch.linspace(
                0.0,
                1.0,
                self.num_quantiles,
                device=logits.device,
                dtype=logits.dtype,
            )

            q0 = torch.quantile(group0, quantile_levels)
            q1 = torch.quantile(group1, quantile_levels)

            fairness = torch.mean(
                torch.abs(q0 - q1) ** self.p
            )

        ce = F.cross_entropy(logits, y_true)

        return (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )