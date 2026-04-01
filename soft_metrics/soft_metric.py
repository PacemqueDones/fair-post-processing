from abc import ABC, abstractmethod

class SoftMetric(ABC):
    name: str

    @abstractmethod
    def __call__(self, y_true, scores):
        """
        Retorna um tensor escalar diferenciável.
        """
        pass