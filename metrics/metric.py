from abc import ABC, abstractmethod

class Metric(ABC):
    name: str
    direction: str   # "max" ou "min"
    type: str        # "performance" ou "fairness"

    @abstractmethod
    def __call__(self, y_true, y_pred, sensitive_attr=None) -> float:
        """
        Calcula o valor da métrica.

        Parâmetros
        ----------
        y_true : torch.Tensor
            Rótulos verdadeiros.
        y_pred : torch.Tensor
            Predições binárias já thresholdadas.
        sensitive_attr : torch.Tensor | None
            Atributo sensível, quando necessário.

        Retorno
        -------
        float
            Valor escalar da métrica.
        """
        pass