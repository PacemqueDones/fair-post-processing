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
    sensitive_features = torch.as_tensor(sensitive_features)

    if sensitive_features.ndim == 1:
        sensitive_features = sensitive_features.reshape(-1, 1)

    for metric in metrics:
        value = metric(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=sensitive_features,
            scores=None,
        )

        results[metric.name] = float(round(value, 4))

    return results