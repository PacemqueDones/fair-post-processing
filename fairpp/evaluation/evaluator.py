import torch

def calculate_metrics(
    y_true,
    y_pred,
    sensitive_features,
    metrics,
    logits=None,
):
    results = {}

    y_true = torch.as_tensor(y_true, dtype=torch.long).reshape(-1)

    y_pred = torch.as_tensor(y_pred, dtype=torch.long).reshape(-1)

    sensitive_attr = torch.as_tensor(sensitive_features)

    if sensitive_attr.ndim == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    if logits is not None:
        logits = torch.as_tensor(
            logits,
            dtype=torch.float32,
        )

        if logits.ndim != 2:
            raise ValueError(
                "logits deve ter shape "
                "(n_samples, n_classes)."
            )
        
        if logits.shape[0] != y_true.shape[0]:
            raise ValueError(
                "logits e y_true devem possuir "
                "o mesmo número de amostras."
            )

    for metric in metrics:
        value = metric(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attr=sensitive_attr,
            logits=logits,
        )

        results[metric.name] = float(round(value, 4))

    return results
