import torch


def calculate_metrics(y_true, y_pred, sensitive_features, metrics):
    results = {}

    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)

    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)

    if not torch.is_tensor(sensitive_features):
        sensitive_features = torch.tensor(sensitive_features)

    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    sensitive_features = sensitive_features.view(-1)

    for metric in metrics:
        value = metric(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=sensitive_features,
            scores=None
        )

        results[metric.name] = float(round(value, 4))

    return results