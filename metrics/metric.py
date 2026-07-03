from abc import ABC, abstractmethod


class Metric(ABC):
    name: str
    direction: str
    type: str

    @abstractmethod
    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ) -> float:
        """Calcula e devolve um único valor escalar."""
        pass
