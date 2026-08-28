from abc import ABC, abstractmethod

import torch


class Objective(ABC):
    """
    Base class for post-processing objectives.

    Every objective returns a list of scalar losses and a
    corresponding list of unique names.
    """

    name: str = None

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[str]]:
        """
        Compute one or more scalar objective losses.

        Parameters
        ----------
        logits
            Model outputs after the smooth decision rule.

        y_true
            Ground-truth labels.

        sensitive_attr
            Sensitive attribute vector or matrix.

        Returns
        -------
        losses
            List of scalar tensors. Each tensor is treated as an
            independent objective by the Jacobian-based optimizer.

        names
            Unique name for each returned loss. The order must match
            the order of ``losses``.
        """
        ...