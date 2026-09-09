import torch
import torch.nn.functional as F

from .objective import Objective
from .reduction import reduce_losses
from .validation import prepare_categorical_sensitive_attributes

_UNSET = object()

class _MarginalGroupFairnessObjective(Objective):
    """Shared machinery for marginal multigroup fairness objectives."""

    valid_reductions = {
        "none",
        "mean",
        "sum",
        "max",
    }

    def _validate_reductions(self):
        if self.class_reduction not in self.valid_reductions:
            raise ValueError(
                "class_reduction must be "
                "'none', 'mean', 'sum', or 'max'."
            )

        if self.group_reduction not in self.valid_reductions:
            raise ValueError(
                "group_reduction must be "
                "'none', 'mean', 'sum', or 'max'."
            )

        if self.attribute_reduction not in self.valid_reductions:
            raise ValueError(
                "attribute_reduction must be "
                "'none', 'mean', 'sum', or 'max'."
            )

        if (
            self.class_reduction == "none"
            and self.group_reduction != "none"
        ):
            raise ValueError(
                "group_reduction must be 'none' "
                "when class_reduction is 'none'."
            )

        if (
            self.group_reduction == "none"
            and self.attribute_reduction != "none"
        ):
            raise ValueError(
                "attribute_reduction must be 'none' "
                "when group_reduction is 'none'."
            )

    def _pairwise_disparity(
        self,
        left,
        right,
    ):
        """
        Compute class-wise disparity between two group mean
        probability vectors.

        Parameters
        ----------
        left, right : torch.Tensor
            Vectors in R^C containing the mean predicted
            probability for each class in two sensitive groups.

        Returns
        -------
        torch.Tensor
            Vector in R^C containing one disparity contribution
            per class.
        """
        difference = left - right

        if self.relaxation == "abs":
            return torch.abs(difference)

        if self.relaxation == "squared":
            return difference.square()

        if self.relaxation == "kl":
            left = left.clamp_min(self.eps)
            right = right.clamp_min(self.eps)

            forward = left * torch.log(left / right)
            reverse = right * torch.log(right / left)

            return forward + reverse

        raise RuntimeError(f"Unknown relaxation: {self.relaxation}")

    def _bernoulli_disparity(
        self,
        left,
        right,
    ):
        """
        Compute disparity between two scalar probabilities.

        Used by multiclass Equality Opportunity, where each
        class is treated conditionally on Y = c.
        """
        difference = left - right

        if self.relaxation == "abs":
            return torch.abs(difference)

        if self.relaxation == "squared":
            return difference.square()

        if self.relaxation == "kl":
            left = torch.clamp(
                left,
                min=self.eps,
                max=1.0 - self.eps,
            )
            right = torch.clamp(
                right,
                min=self.eps,
                max=1.0 - self.eps,
            )

            forward = (
                left * torch.log(left / right)
                + (1.0 - left)
                * torch.log(
                    (1.0 - left)
                    / (1.0 - right)
                )
            )

            reverse = (
                right * torch.log(right / left)
                + (1.0 - right)
                * torch.log(
                    (1.0 - right)
                    / (1.0 - left)
                )
            )

            return forward + reverse

        raise RuntimeError(
            f"Unknown relaxation: {self.relaxation}"
        )

    def _compute_group_means(
        self,
        values,
        attribute,
    ):
        """
        Compute the mean prediction vector for each sensitive group.
        """
        groups = torch.unique(
            attribute,
            sorted=True,
        )

        means = []
        valid_groups = []

        for group in groups:
            group_values = values[attribute == group]

            if group_values.shape[0] == 0:
                continue

            means.append(group_values.mean(dim=0))
            valid_groups.append(group)

        return means, valid_groups

    def _reduce_classes(
        self,
        disparities,
        pair_name,
    ):
        """
        Reduce class-wise disparity contributions.
        """
        losses = [disparities[class_index] for class_index in range(disparities.shape[0])]

        names = [f"{pair_name}_class_{class_index}" for class_index in range(disparities.shape[0])]

        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.class_reduction,
            reduced_name=pair_name,
        )

    def _reduce_groups(
        self,
        means,
        groups,
        attribute_name,
        logits,
    ):
        """
        Compare all pairs of sensitive groups and reduce
        the resulting disparities within one sensitive attribute.
        """
        if len(means) < 2:
            zero = logits.sum() * 0.0

            return [zero], [f"{attribute_name}_insufficient_groups"]

        pair_losses = []
        pair_names = []

        for left_index in range(len(means) - 1):
            for right_index in range(left_index + 1, len(means)):
                left_group = groups[left_index].item()

                right_group = groups[right_index].item()

                pair_name = (
                    f"{attribute_name}_group_"
                    f"{left_group}_vs_{right_group}"
                )

                disparities = self._pairwise_disparity(means[left_index], means[right_index])

                class_losses, class_names = self._reduce_classes(
                    disparities=disparities, 
                    pair_name=pair_name
                    )

                pair_losses.extend(class_losses)
                pair_names.extend(class_names)

        return reduce_losses(
            losses=pair_losses,
            names=pair_names,
            reduction=self.group_reduction,
            reduced_name=attribute_name,
        )

    def _reduce_attributes(
        self,
        losses,
        names,
    ):
        """
        Reduce fairness objectives across sensitive attributes.
        """
        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.attribute_reduction,
            reduced_name=self.name,
        )


class DemographicParityObjective(_MarginalGroupFairnessObjective):
    name = "demographic_parity"

    def __init__(
        self,
        relaxation="abs",
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction="none",
        eps=1e-7,
    ):
        valid_relaxations = {
            "abs",
            "squared",
            "kl",
        }

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be "
                "'abs', 'squared', or 'kl'."
            )

        if relaxation == "kl" and class_reduction != "sum":
            raise ValueError(
                "class_reduction must be 'sum' when "
                "relaxation='kl', because summation over "
                "classes is part of the categorical "
                "Kullback-Leibler divergence."
            )

        self.relaxation = relaxation
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction
        self.eps = eps

        self.name = f"demographic_parity_{relaxation}"

        self._validate_reductions()

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        sensitive_attr, sensitive_indices = prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )

        probabilities = torch.softmax(logits, dim=1)

        losses = []
        names = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]

            attribute_name = (
                f"{self.name}_attribute_"
                f"{original_index}"
            )

            means, groups = self._compute_group_means(
                    values=probabilities,
                    attribute=attribute,
                )

            attribute_losses, attribute_names = self._reduce_groups(
                means=means,
                groups=groups,
                attribute_name=attribute_name,
                logits=logits,
            )

            losses.extend(attribute_losses)

            names.extend(attribute_names)

        return self._reduce_attributes(
            losses=losses,
            names=names,
        )


class EqualityOpportunityObjective(_MarginalGroupFairnessObjective):
    name = "equality_opportunity"

    def __init__(
        self,
        relaxation="abs",
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction="none",
        eps=1e-7,
    ):
        valid_relaxations = {
            "abs",
            "squared",
            "kl",
        }

        if relaxation not in valid_relaxations:
            raise ValueError(
                "relaxation must be "
                "'abs', 'squared', or 'kl'."
            )

        if (
            relaxation == "kl"
            and class_reduction != "sum"
        ):
            raise ValueError(
                "class_reduction must be 'sum' when "
                "relaxation='kl'."
            )

        self.relaxation = relaxation
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction
        self.eps = eps

        self.name = f"equality_opportunity_{relaxation}"

        self._validate_reductions()

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        sensitive_attr, sensitive_indices = prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )
    
        probabilities = torch.softmax(logits, dim=1)

        num_classes = probabilities.shape[1]

        losses = []
        names = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]

            attribute_name = (
                f"{self.name}_attribute_"
                f"{original_index}"
            )

            groups = torch.unique(attribute, sorted=True)

            pair_losses = []
            pair_names = []

            for left_index in range(len(groups) - 1):
                for right_index in range(left_index + 1, len(groups)):
                    left_group = groups[left_index]

                    right_group = groups[right_index]

                    pair_name = (
                        f"{attribute_name}_group_"
                        f"{left_group.item()}_vs_"
                        f"{right_group.item()}"
                    )

                    class_disparities = []

                    for class_index in range(num_classes):
                        class_mask = y_true == class_index

                        left_mask = class_mask & (attribute == left_group)

                        right_mask = class_mask & (attribute == right_group)

                        if left_mask.sum() == 0 or right_mask.sum() == 0:
                            continue

                        left_mean = probabilities[left_mask, class_index].mean()

                        right_mean = probabilities[right_mask, class_index].mean()

                        disparity = self._bernoulli_disparity(left_mean, right_mean)

                        class_disparities.append(disparity)

                    if not class_disparities:
                        continue

                    disparities = torch.stack(class_disparities)

                    class_losses, class_names = self._reduce_classes(
                        disparities=disparities,
                        pair_name=pair_name
                    )

                    pair_losses.extend(class_losses)

                    pair_names.extend(class_names)

            if not pair_losses:
                zero = logits.sum() * 0.0

                attribute_losses = [zero]
                attribute_names = [
                    f"{attribute_name}_"
                    "insufficient_groups"
                ]

            else:
                attribute_losses, attribute_names = reduce_losses(
                    losses=pair_losses,
                    names=pair_names,
                    reduction=(self.group_reduction),
                    reduced_name=attribute_name,
                )

            losses.extend(attribute_losses)

            names.extend(attribute_names)

        return self._reduce_attributes(
            losses=losses,
            names=names,
        )


class WassersteinDemographicParityObjective(Objective):
    """
    Compare group score distributions with a Wasserstein barycenter.

    The objective is computed independently for every predicted class.
    For ``p=1``, barycenter quantiles are weighted medians. For ``p=2``,
    they are weighted means. The returned group losses approximate
    ``W_p^p``; consequently, ``p=2`` returns squared Wasserstein losses.

    Parameters
    ----------
    p : int, default=2
        Wasserstein order. Supported values are 1 and 2.

    num_quantiles : int, default=100
        Number of midpoint quantiles used to approximate the integral
        over the unit interval.

    sensitive_indices : sequence of int or None, default=None
        Sensitive-attribute columns used by the objective.

    grouping : {"marginal", "intersectional"}, default="marginal"
        Whether sensitive attributes are handled separately or combined
        into intersectional groups.

    class_reduction, group_reduction : {"none", "mean", "sum", "max"}
        Reductions over predicted classes and sensitive groups.

    attribute_reduction : {"none", "mean", "sum", "max"}, None, or omitted
        In marginal grouping, omission resolves to ``"none"``. In
        intersectional grouping, omission resolves to ``None`` because
        there is no marginal attribute axis.
    """

    name = "wasserstein_demographic_parity"

    valid_groupings = {
        "marginal",
        "intersectional",
    }

    valid_reductions = {
        "none",
        "mean",
        "sum",
        "max",
    }

    def __init__(
        self,
        p=2,
        num_quantiles=100,
        sensitive_indices=None,
        grouping="marginal",
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction=_UNSET,
    ):
        if p not in {1, 2}:
            raise ValueError("p must be either 1 or 2.")

        if not isinstance(num_quantiles, int) or num_quantiles < 1:
            raise ValueError(
                "num_quantiles must be a positive integer."
            )

        if grouping not in self.valid_groupings:
            raise ValueError(
                "grouping must be 'marginal' or 'intersectional'."
            )

        if attribute_reduction is _UNSET:
            attribute_reduction = (
                "none"
                if grouping == "marginal"
                else None
            )

        self.p = p
        self.num_quantiles = num_quantiles
        self.sensitive_indices = sensitive_indices
        self.grouping = grouping
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction
        self.name = f"wasserstein_demographic_parity_p{p}"

        self._validate_configuration()

    def _validate_configuration(self):
        if self.class_reduction not in self.valid_reductions:
            raise ValueError(
                "class_reduction must be "
                "'none', 'mean', 'sum', or 'max'."
            )

        if self.group_reduction not in self.valid_reductions:
            raise ValueError(
                "group_reduction must be "
                "'none', 'mean', 'sum', or 'max'."
            )

        if self.grouping == "marginal":
            if self.attribute_reduction not in self.valid_reductions:
                raise ValueError(
                    "For grouping='marginal', attribute_reduction "
                    "must be 'none', 'mean', 'sum', or 'max'."
                )
        elif self.attribute_reduction is not None:
            raise ValueError(
                "attribute_reduction does not apply when "
                "grouping='intersectional'. Use None or omit it."
            )

        if (
            self.class_reduction == "none"
            and self.group_reduction != "none"
        ):
            raise ValueError(
                "group_reduction must be 'none' when "
                "class_reduction is 'none'."
            )

        if (
            self.grouping == "marginal"
            and self.group_reduction == "none"
            and self.attribute_reduction != "none"
        ):
            raise ValueError(
                "attribute_reduction must be 'none' when "
                "group_reduction is 'none'."
            )

    def _quantile_levels(self, reference):
        indices = torch.arange(
            self.num_quantiles,
            device=reference.device,
            dtype=reference.dtype,
        )

        return (indices + 0.5) / self.num_quantiles

    def _compute_group_quantiles(
        self,
        probabilities,
        group_codes,
    ):
        groups = torch.unique(group_codes, sorted=True)
        levels = self._quantile_levels(probabilities)

        quantiles = []
        counts = []
        valid_groups = []

        for group in groups:
            group_values = probabilities[group_codes == group]

            if group_values.shape[0] == 0:
                continue

            group_quantiles = torch.quantile(
                group_values,
                levels,
                dim=0,
            ).transpose(0, 1)

            quantiles.append(group_quantiles)
            counts.append(group_values.shape[0])
            valid_groups.append(group)

        if not quantiles:
            return None, None, []

        stacked_quantiles = torch.stack(quantiles, dim=0)
        weights = probabilities.new_tensor(counts)
        weights = weights / weights.sum()

        return stacked_quantiles, weights, valid_groups

    @staticmethod
    def _weighted_median(quantiles, weights):
        """Compute weighted medians along the group axis."""
        sorted_quantiles, order = torch.sort(
            quantiles,
            dim=0,
        )

        expanded_weights = weights[:, None, None].expand_as(
            quantiles
        )
        sorted_weights = torch.gather(
            expanded_weights,
            dim=0,
            index=order,
        )
        cumulative_weights = sorted_weights.cumsum(dim=0)

        median_indices = (
            cumulative_weights >= 0.5
        ).to(torch.int64).argmax(dim=0)

        selected = torch.gather(
            sorted_quantiles,
            dim=0,
            index=median_indices.unsqueeze(0),
        ).squeeze(0)

        selected_cumulative = torch.gather(
            cumulative_weights,
            dim=0,
            index=median_indices.unsqueeze(0),
        ).squeeze(0)

        next_indices = (median_indices + 1).clamp_max(
            quantiles.shape[0] - 1
        )
        next_values = torch.gather(
            sorted_quantiles,
            dim=0,
            index=next_indices.unsqueeze(0),
        ).squeeze(0)

        has_next = median_indices < quantiles.shape[0] - 1
        tied_half = torch.isclose(
            selected_cumulative,
            selected_cumulative.new_tensor(0.5),
        )

        return torch.where(
            tied_half & has_next,
            0.5 * (selected + next_values),
            selected,
        )

    def _compute_barycenter(self, quantiles, weights):
        if self.p == 1:
            return self._weighted_median(
                quantiles=quantiles,
                weights=weights,
            )

        return (
            quantiles * weights[:, None, None]
        ).sum(dim=0)

    def _reduce_classes(
        self,
        distances,
        group_name,
    ):
        losses = [
            distances[class_index]
            for class_index in range(distances.shape[0])
        ]
        names = [
            f"{group_name}_class_{class_index}"
            for class_index in range(distances.shape[0])
        ]

        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.class_reduction,
            reduced_name=group_name,
        )

    def _reduce_groups(
        self,
        quantiles,
        weights,
        group_labels,
        grouping_name,
    ):
        barycenter = self._compute_barycenter(
            quantiles=quantiles,
            weights=weights,
        )

        group_losses = []
        group_names = []

        for group_index, group_label in enumerate(group_labels):
            distances = (
                quantiles[group_index] - barycenter
            ).abs().pow(self.p).mean(dim=1).pow(1.0 / self.p)

            group_name = (
                f"{grouping_name}_group_{group_label}"
            )

            class_losses, class_names = self._reduce_classes(
                distances=distances,
                group_name=group_name,
            )

            group_losses.extend(class_losses)
            group_names.extend(class_names)

        return reduce_losses(
            losses=group_losses,
            names=group_names,
            reduction=self.group_reduction,
            reduced_name=grouping_name,
        )

    @staticmethod
    def _format_scalar_group(group):
        return str(group.item())

    @staticmethod
    def _format_intersectional_group(group_row):
        return "_".join(str(value.item()) for value in group_row)

    def _marginal_losses(
        self,
        probabilities,
        sensitive_attr,
        sensitive_indices,
        logits,
    ):
        losses = []
        names = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            attribute_name = (
                f"{self.name}_attribute_{original_index}"
            )

            quantiles, weights, groups = (
                self._compute_group_quantiles(
                    probabilities=probabilities,
                    group_codes=attribute,
                )
            )

            if quantiles is None or len(groups) < 2:
                losses.append(logits.sum() * 0.0)
                names.append(
                    f"{attribute_name}_insufficient_groups"
                )
                continue

            group_labels = [
                self._format_scalar_group(group)
                for group in groups
            ]
            attribute_losses, attribute_names = (
                self._reduce_groups(
                    quantiles=quantiles,
                    weights=weights,
                    group_labels=group_labels,
                    grouping_name=attribute_name,
                )
            )

            losses.extend(attribute_losses)
            names.extend(attribute_names)

        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.attribute_reduction,
            reduced_name=self.name,
        )

    def _intersectional_losses(
        self,
        probabilities,
        sensitive_attr,
        logits,
    ):
        unique_rows, group_codes = torch.unique(
            sensitive_attr,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        grouping_name = f"{self.name}_intersectional"

        quantiles, weights, groups = (
            self._compute_group_quantiles(
                probabilities=probabilities,
                group_codes=group_codes,
            )
        )

        if quantiles is None or len(groups) < 2:
            zero = logits.sum() * 0.0
            return [zero], [
                f"{grouping_name}_insufficient_groups"
            ]

        group_labels = [
            self._format_intersectional_group(unique_rows[group])
            for group in groups
        ]

        return self._reduce_groups(
            quantiles=quantiles,
            weights=weights,
            group_labels=group_labels,
            grouping_name=grouping_name,
        )

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        sensitive_attr, sensitive_indices = (
            prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )
        )

        probabilities = torch.softmax(logits, dim=1)

        if self.grouping == "marginal":
            return self._marginal_losses(
                probabilities=probabilities,
                sensitive_attr=sensitive_attr,
                sensitive_indices=sensitive_indices,
                logits=logits,
            )

        return self._intersectional_losses(
            probabilities=probabilities,
            sensitive_attr=sensitive_attr,
            logits=logits,
        )

class WassersteinEqualityOpportunityObjective(WassersteinDemographicParityObjective):
    """
    Compare class-conditional group score distributions with a
    Wasserstein barycenter.

    For each class ``c``, this objective compares the distributions
    of the predicted score for class ``c`` among observations that
    satisfy ``Y = c``. The barycenter weights are the empirical group
    proportions inside that class, ``P(S = s | Y = c)``.
    """

    name = "wasserstein_equality_opportunity"

    def __init__(
        self,
        p=2,
        num_quantiles=100,
        sensitive_indices=None,
        grouping="marginal",
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction=_UNSET,
    ):
        super().__init__(
            p=p,
            num_quantiles=num_quantiles,
            sensitive_indices=sensitive_indices,
            grouping=grouping,
            class_reduction=class_reduction,
            group_reduction=group_reduction,
            attribute_reduction=attribute_reduction,
        )

        self.name = f"wasserstein_equality_opportunity_p{p}"

    def _conditional_group_losses(
        self,
        probabilities,
        y_true,
        group_codes,
        group_labels,
        grouping_name,
        logits,
    ):
        losses_by_group = {
            int(group.item()): []
            for group in torch.unique(group_codes, sorted=True)
        }
        names_by_group = {
            group_code: []
            for group_code in losses_by_group
        }

        num_classes = probabilities.shape[1]

        for class_index in range(num_classes):
            class_mask = y_true == class_index

            if class_mask.sum() == 0:
                continue

            class_scores = probabilities[
                class_mask,
                class_index,
            ].unsqueeze(1)
            class_group_codes = group_codes[class_mask]

            quantiles, weights, valid_groups = (
                self._compute_group_quantiles(
                    probabilities=class_scores,
                    group_codes=class_group_codes,
                )
            )

            if quantiles is None or len(valid_groups) < 2:
                continue

            barycenter = self._compute_barycenter(
                quantiles=quantiles,
                weights=weights,
            )

            for local_group_index, group in enumerate(valid_groups):
                group_code = int(group.item())
                distance = (
                    quantiles[local_group_index] - barycenter
                ).abs().pow(self.p).mean()

                group_name = (
                    f"{grouping_name}_group_"
                    f"{group_labels[group_code]}"
                )

                losses_by_group[group_code].append(distance)
                names_by_group[group_code].append(
                    f"{group_name}_class_{class_index}"
                )

        group_losses = []
        group_names = []

        for group_code, class_losses in losses_by_group.items():
            if not class_losses:
                continue

            group_name = (
                f"{grouping_name}_group_"
                f"{group_labels[group_code]}"
            )
            reduced_losses, reduced_names = reduce_losses(
                losses=class_losses,
                names=names_by_group[group_code],
                reduction=self.class_reduction,
                reduced_name=group_name,
            )

            group_losses.extend(reduced_losses)
            group_names.extend(reduced_names)

        if not group_losses:
            zero = logits.sum() * 0.0
            return [zero], [
                f"{grouping_name}_insufficient_groups"
            ]

        return reduce_losses(
            losses=group_losses,
            names=group_names,
            reduction=self.group_reduction,
            reduced_name=grouping_name,
        )

    def _marginal_conditional_losses(
        self,
        probabilities,
        y_true,
        sensitive_attr,
        sensitive_indices,
        logits,
    ):
        losses = []
        names = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            attribute_name = (
                f"{self.name}_attribute_{original_index}"
            )
            groups = torch.unique(attribute, sorted=True)
            group_labels = {
                int(group.item()): self._format_scalar_group(group)
                for group in groups
            }

            attribute_losses, attribute_names = (
                self._conditional_group_losses(
                    probabilities=probabilities,
                    y_true=y_true,
                    group_codes=attribute,
                    group_labels=group_labels,
                    grouping_name=attribute_name,
                    logits=logits,
                )
            )

            losses.extend(attribute_losses)
            names.extend(attribute_names)

        return reduce_losses(
            losses=losses,
            names=names,
            reduction=self.attribute_reduction,
            reduced_name=self.name,
        )

    def _intersectional_conditional_losses(
        self,
        probabilities,
        y_true,
        sensitive_attr,
        logits,
    ):
        unique_rows, group_codes = torch.unique(
            sensitive_attr,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        grouping_name = f"{self.name}_intersectional"
        groups = torch.unique(group_codes, sorted=True)
        group_labels = {
            int(group.item()): self._format_intersectional_group(
                unique_rows[group]
            )
            for group in groups
        }

        return self._conditional_group_losses(
            probabilities=probabilities,
            y_true=y_true,
            group_codes=group_codes,
            group_labels=group_labels,
            grouping_name=grouping_name,
            logits=logits,
        )

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
        base_outputs=None,
        input_type=None,
    ):
        sensitive_attr, sensitive_indices = (
            prepare_categorical_sensitive_attributes(
                sensitive_attr=sensitive_attr,
                num_samples=logits.shape[0],
                sensitive_indices=self.sensitive_indices,
            )
        )

        probabilities = torch.softmax(logits, dim=1)

        if self.grouping == "marginal":
            return self._marginal_conditional_losses(
                probabilities=probabilities,
                y_true=y_true,
                sensitive_attr=sensitive_attr,
                sensitive_indices=sensitive_indices,
                logits=logits,
            )

        return self._intersectional_conditional_losses(
            probabilities=probabilities,
            y_true=y_true,
            sensitive_attr=sensitive_attr,
            logits=logits,
        )