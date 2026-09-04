import torch


def reduce_values(values, reduction):
    """
    Reduz uma lista de valores escalares.

    mean:
        média dos valores

    sum:
        soma dos valores

    max:
        maior valor
    """
    if reduction not in {"mean", "sum", "max"}:
        raise ValueError(
            "reduction deve ser 'mean', 'sum' ou 'max'."
        )

    if not values:
        raise ValueError("A lista de valores não pode estar vazia.")

    values = torch.stack(values)

    if reduction == "mean":
        return values.mean()

    if reduction == "sum":
        return values.sum()

    return values.max()


def reduce_group_rates(group_rates, reduction):
    """
    Calcula a disparidade entre as taxas dos grupos.

    max:
        maior taxa - menor taxa

    mean:
        média das diferenças absolutas entre todos os pares

    sum:
        soma das diferenças absolutas entre todos os pares
    """
    if reduction not in {"mean", "sum", "max"}:
        raise ValueError(
            "reduction deve ser 'mean', 'sum' ou 'max'."
        )

    if group_rates.numel() < 2:
        return group_rates.sum() * 0.0

    if reduction == "max":
        return group_rates.max() - group_rates.min()

    group_rates = torch.sort(group_rates).values
    num_groups = group_rates.numel()

    positions = torch.arange(
        1,
        num_groups + 1,
        device=group_rates.device,
        dtype=group_rates.dtype,
    )

    coefficients = 2 * positions - num_groups - 1
    pairwise_sum = torch.sum(coefficients * group_rates)

    if reduction == "sum":
        return pairwise_sum

    num_pairs = num_groups * (num_groups - 1) / 2

    return pairwise_sum / num_pairs