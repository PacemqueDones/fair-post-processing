import torch
import torch.nn as nn
    
class ThresholdRatioModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs, sensitive_attr=None):
        ratios = probs / (self.thresholds + self.eps)
        return self.alpha * ratios

class ThresholdRatioSiLUModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs, sensitive_attr=None):
        ratios = probs / (self.thresholds + self.eps)
        logits = torch.nn.functional.silu(self.alpha * ratios)
        return logits
    
class ThresholdRatioDGateModel(nn.Module):
    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()
        self.thresholds = nn.Parameter(torch.rand(num_classes))
        self.direction = nn.Parameter(torch.ones(num_classes))
        self.alpha = alpha
        self.eps = eps

    def forward(self, probs, sensitive_attr=None):
        ratios = probs / (self.thresholds + self.eps)
        u = self.alpha * ratios
        gate = self.direction * torch.sigmoid(u)
        logits = gate * u
        return logits

class ThresholdCategoricalAdditiveRatioModel(nn.Module):
    """Modelo de thresholds categóricos aditivos.

    Para cada amostra i e classe c, o threshold latente é

        eta[i, c] = beta0[c] + sum_j beta_j[A[i, j], c],

    em que a categoria 0 de cada atributo é usada como referência e possui
    correção fixa igual a zero. Para um atributo j com K_j categorias, a
    matriz treinável beta_j possui, portanto, shape (K_j - 1, C).

    Os thresholds positivos são obtidos por

        threshold[i, c] = softplus(eta[i, c]) + eps.

    Em seguida, o modelo preserva a essência do ThresholdRatioModel:

        logits[i, c] = alpha * probs[i, c] / threshold[i, c].

    Parameters
    ----------
    num_classes:
        Número de classes C.
    category_sizes:
        Quantidade de categorias de cada atributo sensível. Por exemplo,
        [2, 5] para sexo com 2 categorias e raça com 5 categorias.
    alpha:
        Fator multiplicativo aplicado aos ratios.
    eps:
        Constante numérica para manter os thresholds estritamente positivos.
    initial_threshold:
        Threshold comum usado na inicialização. Todas as categorias começam
        com o mesmo threshold; as diferenças entre grupos são aprendidas.
    """

    def __init__(
        self,
        num_classes,
        category_sizes,
        alpha=10.0,
        eps=1e-8,
        initial_threshold=0.5,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes deve ser maior ou igual a 2.")

        if not category_sizes:
            raise ValueError("category_sizes deve conter ao menos um atributo.")

        if any(int(size) < 2 for size in category_sizes):
            raise ValueError(
                "Cada atributo sensível deve possuir ao menos duas categorias."
            )

        if initial_threshold <= eps:
            raise ValueError("initial_threshold deve ser maior que eps.")

        self.num_classes = int(num_classes)
        self.category_sizes = tuple(int(size) for size in category_sizes)
        self.num_sensitive_attributes = len(self.category_sizes)
        self.alpha = alpha
        self.eps = eps

        # softplus(beta0) + eps = initial_threshold
        target = float(initial_threshold - eps)
        initial_beta0 = torch.log(torch.expm1(torch.tensor(target)))

        # Efeito-base, comum a todas as amostras antes das correções categóricas.
        self.beta0 = nn.Parameter(
            torch.full((self.num_classes,), initial_beta0.item())
        )

        # Para o atributo j, a linha r representa a categoria r + 1.
        # A categoria 0 é a referência e possui correção fixa igual a zero.
        self.category_betas = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(size - 1, self.num_classes)
                )
                for size in self.category_sizes
            ]
        )

    @property
    def num_threshold_parameters(self):
        """Número de parâmetros treináveis usados para formar thresholds."""
        return self.num_classes * (
            1 + sum(size - 1 for size in self.category_sizes)
        )

    def _validate_inputs(self, probs, sensitive_attr):
        if probs.ndim != 2:
            raise ValueError(
                "probs deve possuir shape (num_samples, num_classes)."
            )

        if probs.shape[1] != self.num_classes:
            raise ValueError(
                f"probs possui {probs.shape[1]} classes, mas o modelo foi "
                f"criado com num_classes={self.num_classes}."
            )

        if sensitive_attr.ndim == 1:
            if self.num_sensitive_attributes != 1:
                raise ValueError(
                    "sensitive_attr unidimensional só é válido quando existe "
                    "um único atributo sensível."
                )
            sensitive_attr = sensitive_attr.unsqueeze(1)

        if sensitive_attr.ndim != 2:
            raise ValueError(
                "sensitive_attr deve possuir shape "
                "(num_samples, num_sensitive_attributes)."
            )

        if sensitive_attr.shape[0] != probs.shape[0]:
            raise ValueError(
                "probs e sensitive_attr devem possuir o mesmo número de amostras."
            )

        if sensitive_attr.shape[1] != self.num_sensitive_attributes:
            raise ValueError(
                f"sensitive_attr possui {sensitive_attr.shape[1]} colunas, "
                f"mas eram esperadas {self.num_sensitive_attributes}."
            )

        return sensitive_attr.long()

    def latent_thresholds(self, probs, sensitive_attr):
        """Calcula eta com shape (num_samples, num_classes)."""
        sensitive_attr = self._validate_inputs(probs, sensitive_attr)

        eta = self.beta0.unsqueeze(0).expand(probs.shape[0], -1)

        for attribute_index, (size, beta) in enumerate(
            zip(self.category_sizes, self.category_betas)
        ):
            category_ids = sensitive_attr[:, attribute_index]

            if torch.any(category_ids < 0) or torch.any(category_ids >= size):
                raise ValueError(
                    f"O atributo sensível {attribute_index} deve conter "
                    f"categorias entre 0 e {size - 1}."
                )

            # A primeira linha, correspondente à categoria 0, é fixa em zero.
            reference = torch.zeros(
                1,
                self.num_classes,
                device=beta.device,
                dtype=beta.dtype,
            )
            effects = torch.cat((reference, beta), dim=0)

            eta = eta + effects[category_ids]

        return eta

    def get_thresholds(self, probs, sensitive_attr):
        """Calcula os thresholds positivos de cada amostra e classe."""
        eta = self.latent_thresholds(probs, sensitive_attr)
        return torch.nn.functional.softplus(eta) + self.eps

    def forward(self, probs, sensitive_attr):
        thresholds = self.get_thresholds(probs, sensitive_attr)
        ratios = probs / thresholds
        return self.alpha * ratios
