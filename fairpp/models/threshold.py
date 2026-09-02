import torch
import torch.nn as nn
import torch.nn.functional as F

def _validate_input_type(input_type):
    if input_type not in {"probability", "score"}:
        raise ValueError(
            "input_type deve ser 'probability' ou 'score'."
        )


def _prepare_inputs(inputs, input_type, eps):
    """Converte probabilidades ou scores em log-probabilidades."""
    if input_type == "probability":
        return torch.log(inputs.clamp_min(eps))

    return F.log_softmax(inputs, dim=1)


class AffineModel(nn.Module):
    """
    Transformação afim global por classe:

        Z = alpha * (scale * X + bias)

    Preparação da entrada:
        input_type="probability":
            X = log(inputs.clamp_min(eps))

        input_type="score":
            X = log_softmax(inputs, dim=1)

    Entradas:
        inputs: probabilidades ou scores, shape (n, C).
        Scores incluem logits; valores maiores indicam maior
        preferência pela classe correspondente.

    Saída:
        Scores ajustados, shape (n, C), sem softmax final.
    """

    def __init__(
        self,
        num_classes,
        alpha=1.0,
        eps=1e-8,
        input_type="probability",
    ):
        super().__init__()

        _validate_input_type(input_type)

        self.input_type = input_type

        initial_log_scale = torch.log(torch.expm1(torch.tensor(1.0))).item()

        self.log_scale = nn.Parameter(torch.full((num_classes,), initial_log_scale))
        self.bias = nn.Parameter(torch.zeros(num_classes))

        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        base_logits = _prepare_inputs(
            inputs,
            self.input_type,
            self.eps,
        )

        scale = F.softplus(self.log_scale)

        adjusted_logits = self.alpha * (
            scale * base_logits + self.bias
        )

        return adjusted_logits


class CategoricalAdditiveModel(nn.Module):
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

        base_values =
            log(inputs.clamp_min(eps)), se input_type="probability"
            log_softmax(inputs, dim=1), se input_type="score"

        logits[i] =
            alpha * (
                scale[i] * base_values[i]
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
        input_type="probability",
    ):
        super().__init__()

        _validate_input_type(input_type)

        self.input_type = input_type

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

    def forward(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        scale, bias = self.get_scale_and_bias(inputs, sensitive_attr)

        base_logits = _prepare_inputs(inputs, self.input_type, self.eps)

        adjusted_logits = scale * base_logits + bias

        return self.alpha * adjusted_logits



class CovariateAffineModel(nn.Module):
    """
    Transformação afim condicionada pelas covariáveis:

        scale = softplus(X @ W_s + c_s)
        bias = X @ W_b + c_b

        adjusted_logits = scale * base_logits + bias

    Preparação da entrada:
        probability: log(inputs.clamp_min(eps))
        score: log_softmax(inputs, dim=1)

    Dimensões:
        inputs: (n, C)
        X: (n, d)
        W_s, W_b: (d, C)
        c_s, c_b: (C,)
        saída: (n, C)

    O atributo sensível não participa do forward.
    A inicialização preserva as probabilidades originais,
    salvo a proteção numérica aplicada pelo clamp.
    """

    def __init__(
        self,
        num_classes,
        num_features,
        alpha=1.0,
        eps=1e-8,
        input_type="probability",
    ):
        super().__init__()

        _validate_input_type(input_type)

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError(
                "num_classes deve ser um inteiro maior ou igual a 2."
            )

        if (
            isinstance(num_features, bool)
            or not isinstance(num_features, int)
            or num_features < 1
        ):
            raise ValueError(
                "num_features deve ser um inteiro positivo."
            )

        if not 0 < eps < 1:
            raise ValueError("eps deve estar entre 0 e 1.")

        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.input_type = input_type
        self.eps = eps

        # softplus(initial_scale_bias) = 1
        initial_scale_bias = torch.log(
            torch.expm1(torch.tensor(1.0))
        ).item()

        self.W_s = nn.Parameter(
            torch.zeros(self.num_features, self.num_classes)
        )
        self.c_s = nn.Parameter(
            torch.full((self.num_classes,), initial_scale_bias)
        )

        self.W_b = nn.Parameter(
            torch.zeros(self.num_features, self.num_classes)
        )
        self.c_b = nn.Parameter(
            torch.zeros(self.num_classes)
        )

    def _validate_covariates(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("X deve ser um torch.Tensor.")

        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(
                f"X deve possuir shape (num_samples, {self.num_features})."
            )

        if not torch.is_floating_point(X):
            raise TypeError("X deve possuir dtype de ponto flutuante.")

        if X.device != self.W_s.device:
            raise ValueError(
                "X e os parâmetros do modelo devem estar no mesmo device."
            )

        if X.dtype != self.W_s.dtype:
            raise TypeError(
                "X e os parâmetros do modelo devem possuir o mesmo dtype."
            )

        if not torch.isfinite(X).all():
            raise ValueError("X não pode conter NaN ou infinito.")

    def _validate_inputs(self, inputs, X):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs deve ser um torch.Tensor.")

        if inputs.ndim != 2 or inputs.shape[1] != self.num_classes:
            raise ValueError(
                f"inputs deve possuir shape "
                f"(num_samples, {self.num_classes})."
            )

        if inputs.shape[0] != X.shape[0]:
            raise ValueError(
                "inputs e X devem possuir o mesmo número de amostras."
            )

        if inputs.device != X.device or inputs.dtype != X.dtype:
            raise ValueError(
                "inputs e X devem possuir o mesmo device e dtype."
            )

        if not torch.isfinite(inputs).all():
            raise ValueError("inputs não pode conter NaN ou infinito.")

        if self.input_type == "probability":
            if torch.any(inputs < 0) or torch.any(inputs > 1):
                raise ValueError(
                    "Probabilidades devem estar entre 0 e 1."
                )

            row_sums = inputs.sum(dim=1)

            if not torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError(
                    "As probabilidades devem somar 1 em cada linha."
                )

    def get_scale_and_bias(self, X):
        self._validate_covariates(X)

        latent_scale = X @ self.W_s + self.c_s
        scale = F.softplus(latent_scale)

        # Protege contra underflow numérico em valores muito negativos,
        # sem acrescentar eps à escala nem alterar a inicialização.
        scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)

        bias = X @ self.W_b + self.c_b

        return scale, bias

    def forward(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        scale, bias = self.get_scale_and_bias(X)

        self._validate_inputs(inputs, X)

        base_logits = _prepare_inputs(
            inputs,
            self.input_type,
            self.eps,
        )

        adjusted_logits = scale * base_logits + bias

        return self.alpha * adjusted_logits