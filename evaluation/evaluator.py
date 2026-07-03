import torch

def calculate_metrics(
    y_true,
    y_pred,
    sensitive_features,
    metrics,
):
    results = {}

    y_true = torch.as_tensor(y_true).view(-1)
    y_pred = torch.as_tensor(y_pred).view(-1)
    sensitive_attr = torch.as_tensor(sensitive_features)

    if sensitive_attr.ndim == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    for metric in metrics:
        value = metric(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=sensitive_attr,
            logits=None,
        )

        results[metric.name] = float(round(value, 4))

    return results
