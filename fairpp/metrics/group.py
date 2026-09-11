import torch

from .metric import Metric
from .reduction import reduce_group_rates, reduce_values
from .validation import prepare_sensitive_attributes


class DemographicParityMetric(Metric):
    name = "ddp"
    direction = "min"
    type = "fairness"

    def __init__(
        self,
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction="max",
        name=None,
    ):
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction

        if name is not None:
            self.name = name
        elif sensitive_indices is not None and len(sensitive_indices) == 1:
            self.name = f"ddp_attribute_{sensitive_indices[0]}"

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        sensitive_attr = sensitive_attr.to(y_pred.device)

        classes = torch.unique(y_pred, sorted=True)
        attribute_disparities = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            class_disparities = []

            for class_label in classes:
                class_predictions = (y_pred == class_label).float()
                group_rates = []

                for group in torch.unique(attribute, sorted=True):
                    group_mask = attribute == group

                    group_rate = class_predictions[group_mask].mean()
                    group_rates.append(group_rate)

                group_rates = torch.stack(group_rates)

                class_disparity = reduce_group_rates(
                    group_rates=group_rates,
                    reduction=self.group_reduction,
                )

                class_disparities.append(class_disparity)

            attribute_disparity = reduce_values(
                values=class_disparities,
                reduction=self.class_reduction,
            )

            attribute_disparities.append(attribute_disparity)

        total_disparity = reduce_values(
            values=attribute_disparities,
            reduction=self.attribute_reduction,
        )

        return total_disparity.item()


class EqualityOpportunityMetric(Metric):
    name = "deo"
    direction = "min"
    type = "fairness"

    def __init__(
        self,
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        attribute_reduction="max",
        name=None,
    ):
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction
        self.attribute_reduction = attribute_reduction

        if name is not None:
            self.name = name
        elif sensitive_indices is not None and len(sensitive_indices) == 1:
            self.name = f"deo_attribute_{sensitive_indices[0]}"

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        y_true = y_true.to(y_pred.device)
        sensitive_attr = sensitive_attr.to(y_pred.device)

        classes = torch.unique(y_true, sorted=True)
        attribute_disparities = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            class_disparities = []

            for class_label in classes:
                class_mask = y_true == class_label
                group_rates = []

                for group in torch.unique(attribute, sorted=True):
                    group_mask = (attribute == group) & class_mask

                    if group_mask.sum() == 0:
                        continue

                    true_positive_rate = (
                        y_pred[group_mask] == class_label
                    ).float().mean()

                    group_rates.append(true_positive_rate)

                if len(group_rates) < 2:
                    class_disparity = y_pred.float().sum() * 0.0
                else:
                    group_rates = torch.stack(group_rates)

                    class_disparity = reduce_group_rates(
                        group_rates=group_rates,
                        reduction=self.group_reduction,
                    )

                class_disparities.append(class_disparity)

            attribute_disparity = reduce_values(
                values=class_disparities,
                reduction=self.class_reduction,
            )

            attribute_disparities.append(attribute_disparity)

        total_disparity = reduce_values(
            values=attribute_disparities,
            reduction=self.attribute_reduction,
        )

        return total_disparity.item()

import torch

from .metric import Metric
from .reduction import reduce_group_rates, reduce_values
from .validation import prepare_sensitive_attributes


class IntersectionalDemographicParityMetric(Metric):
    """
    Demographic parity across intersectional sensitive groups.

    Each distinct combination of sensitive attributes defines one
    intersectional group.

    For example, if the sensitive attributes are sex and race, the
    groups are combinations such as:

        (Female, White)
        (Female, Black)
        (Male, White)
        (Male, Black)

    rather than sex and race being evaluated separately.
    """

    name = "intersectional_ddp"
    direction = "min"
    type = "fairness"

    def __init__(
        self,
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        name=None,
    ):
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction

        if name is not None:
            self.name = name

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        sensitive_attr = sensitive_attr.to(y_pred.device)

        # ---------------------------------------------------------
        # 1. Form the intersectional groups.
        #
        # Each row of sensitive_attr is the sensitive profile of
        # one individual. Equal rows therefore belong to the same
        # intersectional group.
        #
        # Example:
        #
        #   sex  race
        #    0     0   -> group 0
        #    0     1   -> group 1
        #    1     0   -> group 2
        #    0     0   -> group 0
        #
        # group_ids contains the intersectional group assigned
        # to each sample.
        # ---------------------------------------------------------
        intersectional_groups, group_ids = torch.unique(
            sensitive_attr,
            dim=0,
            sorted=True,
            return_inverse=True,
        )

        # ---------------------------------------------------------
        # 2. Identify the predicted classes.
        # ---------------------------------------------------------
        classes = torch.unique(y_pred, sorted=True)

        class_disparities = []

        # ---------------------------------------------------------
        # 3. For each class, calculate its prediction rate inside
        #    every intersectional group.
        #
        # For class c and intersectional group g:
        #
        #   rate(g, c) = P(y_pred = c | G = g)
        # ---------------------------------------------------------
        for class_label in classes:
            class_predictions = (y_pred == class_label).float()

            group_rates = []

            for group_id in range(intersectional_groups.shape[0]):
                group_mask = group_ids == group_id

                group_rate = class_predictions[group_mask].mean()

                group_rates.append(group_rate)

            group_rates = torch.stack(group_rates)

            # -----------------------------------------------------
            # 4. Measure the disparity between intersectional
            #    groups for the current class.
            # -----------------------------------------------------
            class_disparity = reduce_group_rates(
                group_rates=group_rates,
                reduction=self.group_reduction,
            )

            class_disparities.append(class_disparity)

        # ---------------------------------------------------------
        # 5. Reduce the disparities obtained for all classes into
        #    the final intersectional demographic parity metric.
        # ---------------------------------------------------------
        total_disparity = reduce_values(
            values=class_disparities,
            reduction=self.class_reduction,
        )

        return total_disparity.item()


class IntersectionalEqualityOpportunityMetric(Metric):
    """
    Equality of opportunity across intersectional sensitive groups.

    Each distinct combination of sensitive attributes defines one
    intersectional group.

    For each class, the true positive rate is calculated separately
    for every intersectional group, and the disparity between these
    rates is measured.
    """

    name = "intersectional_deo"
    direction = "min"
    type = "fairness"

    def __init__(
        self,
        sensitive_indices=None,
        class_reduction="mean",
        group_reduction="max",
        name=None,
    ):
        self.sensitive_indices = sensitive_indices
        self.class_reduction = class_reduction
        self.group_reduction = group_reduction

        if name is not None:
            self.name = name

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        y_true = y_true.to(y_pred.device)
        sensitive_attr = sensitive_attr.to(y_pred.device)

        # ---------------------------------------------------------
        # 1. Form the intersectional groups.
        #
        # Each row of sensitive_attr represents the sensitive
        # profile of one individual. Equal rows therefore belong
        # to the same intersectional group.
        #
        # Example:
        #
        #   sex  race
        #    0     0   -> group 0
        #    0     1   -> group 1
        #    1     0   -> group 2
        #    0     0   -> group 0
        #
        # group_ids contains the intersectional group assigned
        # to each sample.
        # ---------------------------------------------------------
        intersectional_groups, group_ids = torch.unique(
            sensitive_attr,
            dim=0,
            sorted=True,
            return_inverse=True,
        )

        # ---------------------------------------------------------
        # 2. Identify the true classes.
        # ---------------------------------------------------------
        classes = torch.unique(y_true, sorted=True)

        class_disparities = []

        # ---------------------------------------------------------
        # 3. For each class, restrict the samples to individuals
        #    whose true label is that class.
        #
        # Then calculate the true positive rate inside every
        # intersectional group:
        #
        #   TPR(g, c) = P(y_pred = c | y_true = c, G = g)
        # ---------------------------------------------------------
        for class_label in classes:
            class_mask = y_true == class_label

            group_rates = []

            for group_id in range(intersectional_groups.shape[0]):
                group_mask = (group_ids == group_id) & class_mask

                # The intersectional group may exist in the dataset
                # but contain no samples from the current class.
                if group_mask.sum() == 0:
                    continue

                true_positive_rate = (
                    y_pred[group_mask] == class_label
                ).float().mean()

                group_rates.append(true_positive_rate)

            # -----------------------------------------------------
            # 4. At least two intersectional groups are required
            #    to define a disparity for the current class.
            # -----------------------------------------------------
            if len(group_rates) < 2:
                class_disparity = y_pred.float().sum() * 0.0

            else:
                group_rates = torch.stack(group_rates)

                class_disparity = reduce_group_rates(
                    group_rates=group_rates,
                    reduction=self.group_reduction,
                )

            class_disparities.append(class_disparity)

        # ---------------------------------------------------------
        # 5. Reduce the disparities obtained for all classes into
        #    the final intersectional equality opportunity metric.
        # ---------------------------------------------------------
        total_disparity = reduce_values(
            values=class_disparities,
            reduction=self.class_reduction,
        )

        return total_disparity.item()