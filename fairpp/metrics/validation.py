import torch


def prepare_sensitive_attributes(
    sensitive_attr,
    num_samples,
    sensitive_indices=None,
):
    """Valida e seleciona as colunas sensíveis usadas pela métrica."""
    if sensitive_attr is None:
        raise ValueError("sensitive_attr deve ser fornecido.")

    sensitive_attr = torch.as_tensor(sensitive_attr)

    if sensitive_attr.ndim == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    if sensitive_attr.ndim != 2:
        raise ValueError(
            "sensitive_attr deve ter shape "
            "(n_samples,) ou (n_samples, n_attributes)."
        )

    if sensitive_attr.shape[0] != num_samples:
        raise ValueError(
            "sensitive_attr e y_pred devem ter o mesmo número de amostras."
        )

    num_attributes = sensitive_attr.shape[1]

    if sensitive_indices is None:
        sensitive_indices = list(range(num_attributes))
    else:
        sensitive_indices = list(sensitive_indices)

    if not sensitive_indices:
        raise ValueError("sensitive_indices não pode ser vazio.")

    for index in sensitive_indices:
        if index < 0 or index >= num_attributes:
            raise ValueError(
                f"Índice sensível inválido: {index}. "
                f"Existem {num_attributes} atributos sensíveis."
            )

    selected = sensitive_attr[:, sensitive_indices]

    return selected, sensitive_indices
