from abc import ABC, abstractmethod
import torch


class Objective(ABC):
    """
    Classe base para todas as funções objetivo do pós-processador.
    """

    name: str = None

    @abstractmethod
    def __call__(
        self,
        scores: torch.Tensor,
        y_true: torch.Tensor,
        sensitive_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula o valor da função objetivo.

        Parameters
        ----------
        scores : torch.Tensor
            Saída do modelo após aplicação da regra de decisão suave.

        y_true : torch.Tensor
            Rótulos verdadeiros.

        sensitive_attr : torch.Tensor
            Atributo sensível.

        Returns
        -------
        torch.Tensor
            Valor escalar da função objetivo.
        """
        ...