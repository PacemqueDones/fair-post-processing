import torch


def reduce_losses(
    losses,
    names,
    reduction,
    reduced_name,
):
    valid_reductions = {
        "none",
        "mean",
        "sum",
        "max",
    }

    if reduction not in valid_reductions:
        raise ValueError(
            "reduction must be 'none', 'mean', "
            "'sum', or 'max'."
        )

    if len(losses) != len(names):
        raise ValueError(
            "losses and names must have the same length."
        )

    if not losses:
        raise ValueError(
            "At least one loss must be provided."
        )

    if reduction == "none":
        return losses, names

    loss_tensor = torch.stack(losses)

    if reduction == "mean":
        reduced_loss = loss_tensor.mean()

    elif reduction == "sum":
        reduced_loss = loss_tensor.sum()

    else:
        reduced_loss = loss_tensor.max()

    return (
        [reduced_loss],
        [f"{reduced_name}_{reduction}"],
    )