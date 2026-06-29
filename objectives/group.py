import torch
import torch.nn.functional as F

from .objective import Objective
from .reduction import reduce_losses
from .validation import (
    prepare_categorical_sensitive_attributes,
)

class _MarginalMultigroupObjective(Objective):
    """Shared machinery for marginal multigroup fairness objectives."""

    valid_reductions = {
        "none",
        "mean",
        "sum",
        "max",
    }

    def _validate_reductions(self):
        if self.group_reduction not in self.valid_reductions:
            raise ValueError(
                "group_reduction must be 'none', 'mean', "
                "'sum', or 'max'."
            )

        if self.attribute_reduction not in self.valid_reductions:
            raise ValueError(
                "attribute_reduction must be 'none', "
                "'mean', 'sum', or 'max'."
            )

        if (
            self.group_reduction == "none"
            and self.attribute_reduction != "none"
        ):
            raise ValueError(
                "attribute_reduction must be 'none' when "
                "group_reduction is 'none', because each "
                "attribute returns a different number of "
                "pairwise losses."
            )

    def _kl_bern(self, p0, p1):
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

    def _pairwise_disparity(self, left, right):
        if self.relaxation == "mean":
            return torch.abs(left - right)

        return (
            self._kl_bern(left, right)
            + self._kl_bern(right, left)
        )

    def _compute_group_means(
        self,
        values,
        attribute,
    ):
        groups = torch.unique(attribute, sorted=True)
        means = []
        valid_groups = []

        for group in groups:
            group_values = values[attribute == group]

            if group_values.numel() == 0:
                continue

            means.append(group_values.mean())
            valid_groups.append(group)

        return means, valid_groups

    def _reduce_within_attribute(
        self,
        means,
        groups,
        attribute_name,
        logits,
    ):
        if len(means) < 2:
            zero = logits.sum() * 0.0
            return [zero], [f"{attribute_name}_insufficient_groups"]

        # For absolute differences between scalar means:
        # max_{a,b} |mu_a - mu_b| = max(mu) - min(mu).
        # This avoids constructing O(K^2) pairs.
        if (
            self.relaxation == "mean"
            and self.group_reduction == "max"
        ):
            means_tensor = torch.stack(means)
            disparity = (
                means_tensor.max()
                - means_tensor.min()
            )
            return [disparity], [f"{attribute_name}_max"]

        pair_losses = []
        pair_names = []

        for left_index in range(len(means) - 1):
            for right_index in range(
                left_index + 1,
                len(means),
            ):
                left_group = groups[left_index].item()
                right_group = groups[right_index].item()

                pair_losses.append(
                    self._pairwise_disparity(
                        means[left_index],
                        means[right_index],
                    )
                )
                pair_names.append(
                    f"{attribute_name}_group_"
                    f"{left_group}_vs_{right_group}"
                )

        return reduce_losses(
            losses=pair_losses,
            names=pair_names,
            reduction=self.group_reduction,
            reduced_name=attribute_name,
        )

    def _add_weights_and_ce(
        self,
        losses,
        ce,
    ):
        return [
            self.fairness_weight * loss
            + self.ce_weight * ce
            for loss in losses
        ]

    def _finalize_attributes(
        self,
        losses,
        names,
    ):
        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.attribute_reduction,
            reduced_name=self.name,
        )


class DemographicParityObjective(_MarginalMultigroupObjective):
    name = "demographic_parity"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0.0,
        relaxation="mean",
        sensitive_indices=None,
        group_reduction="max",
        attribute_reduction='none',
        eps=1e-7,
    ):
        valid_relaxations = {"mean", "kl"}

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be either 'mean' or 'kl'."
            )

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.relaxation = relaxation
        self.sensitive_indices = sensitive_indices
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction
        self.eps = eps
        self.name = f"demographic_parity_{relaxation}"

        self._validate_reductions()

    def __call__(self, logits, y_true, sensitive_attr):
        sensitive_attr, sensitive_indices = (
            prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )
        )

        preds_pos = torch.softmax(logits, dim=1)[:, 1]

        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, y_true)
        else:
            ce = logits.sum() * 0.0

        losses = []
        names = []

        for local_index, original_index in enumerate(
            sensitive_indices
        ):
            attribute = sensitive_attr[:, local_index]
            attribute_name = (
                f"{self.name}_attribute_{original_index}"
            )

            means, groups = self._compute_group_means(
                values=preds_pos,
                attribute=attribute,
            )

            attribute_losses, attribute_names = (
                self._reduce_within_attribute(
                    means=means,
                    groups=groups,
                    attribute_name=attribute_name,
                    logits=logits,
                )
            )

            losses.extend(
                self._add_weights_and_ce(
                    losses=attribute_losses,
                    ce=ce,
                )
            )
            names.extend(attribute_names)

        return self._finalize_attributes(
            losses=losses,
            names=names,
        )


class EqualityOpportunityObjective(_MarginalMultigroupObjective):
    name = "equality_opportunity"

    def __init__(
        self,
        fairness_weight=1.0,
        ce_weight=0.0,
        relaxation="mean",
        sensitive_indices=None,
        group_reduction="max",
        attribute_reduction="none",
        positive_label=1,
        eps=1e-7,
    ):
        valid_relaxations = {"mean", "kl"}

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be either 'mean' or 'kl'."
            )


        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.relaxation = relaxation
        self.sensitive_indices = sensitive_indices
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction
        self.positive_label = positive_label
        self.eps = eps
        self.name = f"equality_opportunity_{relaxation}"

        self._validate_reductions()

    def __call__(self, logits, y_true, sensitive_attr):
        sensitive_attr, sensitive_indices = (
            prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )
        )

        preds_pos = torch.softmax(logits, dim=1)[:, 1]
        positive_mask = y_true == self.positive_label

        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, y_true)
        else:
            ce = logits.sum() * 0.0

        losses = []
        names = []

        for local_index, original_index in enumerate(
            sensitive_indices
        ):
            attribute = sensitive_attr[:, local_index]
            attribute_name = (
                f"{self.name}_attribute_{original_index}"
            )

            conditional_values = preds_pos[positive_mask]
            conditional_attribute = attribute[positive_mask]

            means, groups = self._compute_group_means(
                values=conditional_values,
                attribute=conditional_attribute,
            )

            attribute_losses, attribute_names = (
                self._reduce_within_attribute(
                    means=means,
                    groups=groups,
                    attribute_name=attribute_name,
                    logits=logits,
                )
            )

            losses.extend(
                self._add_weights_and_ce(
                    losses=attribute_losses,
                    ce=ce,
                )
            )
            names.extend(attribute_names)

        return self._finalize_attributes(
            losses=losses,
            names=names,
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