import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LogitAffineModel(nn.Module):
    """
    Transformação afim no espaço dos logits:

        Z = A * log(P) + B

    em que:
        P: probabilidades originais, shape (n, C)
        A: escalas positivas por classe, shape (C,)
        B: deslocamentos por classe, shape (C,)
        Z: logits ajustados, shape (n, C)
    """

    def __init__(self, num_classes, alpha=10.0, eps=1e-8):
        super().__init__()

        self.log_scale = nn.Parameter(torch.zeros(num_classes))

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.alpha = alpha

        self.eps = eps

    def forward(self, probs, sensitive_attr=None):
        base_logits = torch.log(probs.clamp_min(self.eps))

        scale = torch.exp(self.log_scale)

        adjusted_logits = self.alpha * (scale * base_logits+ self.bias)

        return adjusted_logits


class LogitCategoricalAdditiveModel(nn.Module):
    """Transformação afim aditiva no espaço dos logits.

    Para cada amostra i:

        latent_scale[i] =
            gamma0
            + sum_j gamma_j[sensitive_attr[i, j]]

        scale[i] =
            softplus(latent_scale[i]) + eps

        bias[i] =
            beta0
            + sum_j beta_j[sensitive_attr[i, j]]

        logits[i] =
            alpha * (
                scale[i] * log(probs[i])
                + bias[i]
            )

    A categoria 0 de cada atributo sensível é usada como referência.
    Sua correção é fixa e igual a zero.
    """

    def __init__(
        self,
        num_classes,
        category_sizes,
        alpha=1.0,
        eps=1e-8,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes deve ser maior ou igual a 2."
            )

        if not category_sizes:
            raise ValueError(
                "category_sizes deve conter ao menos um atributo."
            )

        if any(int(size) < 2 for size in category_sizes):
            raise ValueError(
                "Cada atributo sensível deve possuir "
                "ao menos duas categorias."
            )

        self.num_classes = int(num_classes)
        self.category_sizes = tuple(int(size) for size in category_sizes)
        self.num_sensitive_attributes = len(self.category_sizes)

        self.alpha = alpha
        self.eps = eps

        # softplus(initial_gamma0) + eps = 1
        target_scale = torch.tensor(1.0 - eps)
        initial_gamma0 = torch.log(torch.expm1(target_scale))

        # Escala-base por classe.
        self.gamma0 = nn.Parameter(torch.full((self.num_classes,), initial_gamma0.item())
        )

        # Deslocamento-base por classe.
        self.beta0 = nn.Parameter(torch.zeros(self.num_classes))

        # Correções categóricas da escala.
        self.category_gammas = nn.ParameterList([nn.Parameter(torch.zeros(size - 1, self.num_classes))for size in self.category_sizes])

        # Correções categóricas do deslocamento.
        self.category_betas = nn.ParameterList([nn.Parameter(torch.zeros(size - 1, self.num_classes)) for size in self.category_sizes])

    def _validate_inputs(self, probs, sensitive_attr):
        if probs.ndim != 2:
            raise ValueError(
                "probs deve possuir shape "
                "(num_samples, num_classes)."
            )

        if probs.shape[1] != self.num_classes:
            raise ValueError(
                f"probs possui {probs.shape[1]} classes, "
                f"mas o modelo possui {self.num_classes}."
            )

        if sensitive_attr.ndim == 1:
            if self.num_sensitive_attributes != 1:
                raise ValueError(
                    "sensitive_attr unidimensional só é válido "
                    "para um único atributo sensível."
                )

            sensitive_attr = sensitive_attr.unsqueeze(1)

        if sensitive_attr.ndim != 2:
            raise ValueError(
                "sensitive_attr deve possuir shape "
                "(num_samples, num_sensitive_attributes)."
            )

        if sensitive_attr.shape[0] != probs.shape[0]:
            raise ValueError(
                "probs e sensitive_attr devem possuir "
                "o mesmo número de amostras."
            )

        if (sensitive_attr.shape[1] != self.num_sensitive_attributes):
            raise ValueError(
                f"Esperados {self.num_sensitive_attributes} "
                "atributos sensíveis."
            )

        return sensitive_attr.long()

    def _category_effects(
        self,
        sensitive_attr,
        parameter_list,
    ):
        """Soma os efeitos categóricos de todos os atributos."""

        num_samples = sensitive_attr.shape[0]

        total_effect = torch.zeros(num_samples, self.num_classes, device=sensitive_attr.device, dtype=parameter_list[0].dtype)

        for attribute_index, (size, parameters) in enumerate(zip(self.category_sizes, parameter_list)):
            category_ids = sensitive_attr[:, attribute_index]

            if (torch.any(category_ids < 0) or torch.any(category_ids >= size)):
                raise ValueError(
                    f"O atributo {attribute_index} deve conter "
                    f"categorias entre 0 e {size - 1}."
                )

            reference = torch.zeros(1, self.num_classes, device=parameters.device, dtype=parameters.dtype)

            effects = torch.cat((reference, parameters), dim=0, )

            total_effect = (total_effect + effects[category_ids])

        return total_effect

    def get_scale_and_bias(
        self,
        probs,
        sensitive_attr,
    ):
        sensitive_attr = self._validate_inputs(probs, sensitive_attr)

        scale_effects = self._category_effects(sensitive_attr, self.category_gammas)

        bias_effects = self._category_effects(sensitive_attr, self.category_betas)

        latent_scale = (self.gamma0.unsqueeze(0) + scale_effects)

        scale = F.softplus(latent_scale) + self.eps

        bias = (self.beta0.unsqueeze(0) + bias_effects)

        return scale, bias

    def forward(self, probs, sensitive_attr):
        scale, bias = self.get_scale_and_bias(probs, sensitive_attr)

        base_logits = torch.log(probs.clamp_min(self.eps))

        adjusted_logits = (scale * base_logits + bias)

        return self.alpha * adjusted_logits