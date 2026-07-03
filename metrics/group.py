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
        within_attribute_reduction="max",
        across_attribute_reduction="max",
        name=None,
    ):
        
        self.sensitive_indices = sensitive_indices
        self.within_attribute_reduction = (
            within_attribute_reduction
        )
        self.across_attribute_reduction = (
            across_attribute_reduction
        )

        if name is not None:
            self.name = name
        elif sensitive_indices is not None and len(sensitive_indices) == 1:
            self.name = f"ddp_attribute_{sensitive_indices[0]}"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        sensitive_attr = sensitive_attr.to(y_pred.device)
        attribute_disparities = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            group_rates = []

            for group in torch.unique(attribute, sorted=True):
                group_mask = attribute == group
                group_rate = y_pred[group_mask].float().mean()
                group_rates.append(group_rate)

            group_rates = torch.stack(group_rates)

            disparity = reduce_group_rates(
                group_rates=group_rates,
                reduction=self.within_attribute_reduction,
            )

            attribute_disparities.append(disparity)

        total_disparity = reduce_values(
            values=attribute_disparities,
            reduction=self.across_attribute_reduction,
        )

        return total_disparity.item()


class EqualityOpportunityMetric(Metric):
    name = "deo"
    direction = "min"
    type = "fairness"

    def __init__(
        self,
        sensitive_indices=None,
        within_attribute_reduction="max",
        across_attribute_reduction="max",
        positive_label=1,
        name=None,
    ):
        self.sensitive_indices = sensitive_indices
        self.within_attribute_reduction = (
            within_attribute_reduction
        )
        self.across_attribute_reduction = (
            across_attribute_reduction
        )
        self.positive_label = positive_label

        if name is not None:
            self.name = name
        elif sensitive_indices is not None and len(sensitive_indices) == 1:
            self.name = f"deo_attribute_{sensitive_indices[0]}"

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        y_true = torch.as_tensor(y_true).view(-1)
        y_pred = torch.as_tensor(y_pred).view(-1)

        sensitive_attr, sensitive_indices = prepare_sensitive_attributes(
            sensitive_attr=sensitive_attr,
            num_samples=y_pred.shape[0],
            sensitive_indices=self.sensitive_indices,
        )

        y_true = y_true.to(y_pred.device)
        sensitive_attr = sensitive_attr.to(y_pred.device)

        positive_mask = y_true == self.positive_label
        attribute_disparities = []

        for local_index, original_index in enumerate(sensitive_indices):
            attribute = sensitive_attr[:, local_index]
            group_rates = []

            for group in torch.unique(attribute, sorted=True):
                group_mask = (attribute == group) & positive_mask

                if group_mask.sum() == 0:
                    continue

                true_positive_rate = y_pred[group_mask].float().mean()
                group_rates.append(true_positive_rate)

            if len(group_rates) < 2:
                disparity = y_pred.float().sum() * 0.0
            else:
                group_rates = torch.stack(group_rates)

                disparity = reduce_group_rates(
                    group_rates=group_rates,
                    reduction=self.within_attribute_reduction,
                )

            attribute_disparities.append(disparity)

        total_disparity = reduce_values(
            values=attribute_disparities,
            reduction=self.across_attribute_reduction,
        )

        return total_disparity.item()
