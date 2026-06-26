import torch


def prepare_binary_sensitive_attributes(
    sensitive_attr,
    num_samples,
    sensitive_indices=None,
):
    if sensitive_attr.ndim == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    if sensitive_attr.ndim != 2:
        raise ValueError(
            "sensitive_attr must have shape "
            "(n_samples,) or "
            "(n_samples, n_sensitive_attributes). "
            f"Received shape: {tuple(sensitive_attr.shape)}."
        )

    if sensitive_attr.shape[0] != num_samples:
        raise ValueError(
            "sensitive_attr must contain the same number "
            "of samples as logits. "
            f"Received {sensitive_attr.shape[0]} sensitive "
            f"samples and {num_samples} logits."
        )

    num_attributes = sensitive_attr.shape[1]

    if sensitive_indices is None:
        sensitive_indices = list(
            range(num_attributes)
        )

    if not sensitive_indices:
        raise ValueError(
            "sensitive_indices cannot be empty."
        )

    invalid_indices = [
        index
        for index in sensitive_indices
        if index < 0 or index >= num_attributes
    ]

    if invalid_indices:
        raise ValueError(
            "Invalid sensitive attribute indices: "
            f"{invalid_indices}. "
            f"Available indices are "
            f"{list(range(num_attributes))}."
        )

    selected_sensitive_attr = sensitive_attr[
        :,
        sensitive_indices,
    ]

    invalid_values = selected_sensitive_attr[
        (selected_sensitive_attr != 0)
        & (selected_sensitive_attr != 1)
    ]

    if invalid_values.numel() > 0:
        unique_invalid_values = torch.unique(
            invalid_values
        )

        raise ValueError(
            "All selected sensitive attributes must be "
            "binary and encoded with values 0 and 1. "
            f"Found invalid values: "
            f"{unique_invalid_values.detach().cpu().tolist()}."
        )

    return (
        selected_sensitive_attr,
        sensitive_indices,
    )