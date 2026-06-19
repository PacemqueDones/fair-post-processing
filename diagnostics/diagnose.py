import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================================================
# Conversão segura para JSON
# =========================================================

def to_jsonable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            to_jsonable(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            to_jsonable(v)
            for v in obj
        ]

    return obj


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(data),
            f,
            ensure_ascii=False,
            indent=4
        )


def save_matrix_csv(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = matrix["columns"]
    indices = matrix["indices"]
    data = matrix["data"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(columns) + "\n")

        for idx, row in zip(indices, data):
            values = [str(idx)]

            for value in row:
                if value is None:
                    values.append("")
                else:
                    values.append(str(value))

            f.write(",".join(values) + "\n")


def matrix_json(columns, indices, data):
    return {
        "columns": columns,
        "indices": indices,
        "data": data,
    }


# =========================================================
# Matrizes básicas do histórico
# =========================================================

def history_points_matrix(post):
    columns = post.metric_names_ or []

    indices = []
    data = []

    for record in post.history_:
        indices.append(int(record.get("epoch")))

        data.append([
            float(value)
            for value in record.get("point", [])
        ])

    return matrix_json(columns, indices, data)


def history_metrics_matrix(post):
    if len(post.history_) == 0:
        return matrix_json([], [], [])

    columns = list(post.history_[0].get("metrics", {}).keys())

    indices = []
    data = []

    for record in post.history_:
        metrics = record.get("metrics", {})

        indices.append(int(record.get("epoch")))

        data.append([
            float(metrics[name])
            for name in columns
        ])

    return matrix_json(columns, indices, data)


def history_losses_matrix(post):
    if len(post.history_) == 0:
        return matrix_json([], [], [])

    columns = list(post.history_[0].get("losses", {}).keys())

    indices = []
    data = []

    for record in post.history_:
        losses = record.get("losses", {})

        indices.append(int(record.get("epoch")))

        data.append([
            float(losses[name])
            for name in columns
        ])

    return matrix_json(columns, indices, data)


def history_thresholds_matrix(post):
    if len(post.history_) == 0:
        return matrix_json([], [], [])

    first_thresholds = post.history_[0].get("thresholds")

    if first_thresholds is None:
        return matrix_json([], [], [])

    if torch.is_tensor(first_thresholds):
        first_thresholds = first_thresholds.detach().cpu().tolist()

    n_thresholds = len(first_thresholds)

    columns = [
        f"threshold_{j}"
        for j in range(n_thresholds)
    ]

    indices = []
    data = []

    for record in post.history_:
        thresholds = record.get("thresholds")

        if torch.is_tensor(thresholds):
            thresholds = thresholds.detach().cpu().tolist()

        indices.append(int(record.get("epoch")))

        data.append([
            float(value)
            for value in thresholds
        ])

    return matrix_json(columns, indices, data)


def history_diagnostics(post):
    rows = []

    for record in post.history_:
        diagnostics = record.get("diagnostics")

        if diagnostics is None:
            continue

        row = {
            "epoch": record.get("epoch"),
            **to_jsonable(diagnostics),
        }

        rows.append(row)

    return rows


# =========================================================
# Pareto
# =========================================================

def normalize_indices(indices):
    if indices is None:
        return []

    if torch.is_tensor(indices):
        return indices.detach().cpu().tolist()

    if isinstance(indices, np.ndarray):
        return indices.tolist()

    return list(indices)


def pareto_points_with_epochs_matrix(post):
    columns = post.metric_names_ or []

    pareto_indices = normalize_indices(
        getattr(post, "pareto_indices_", None)
    )

    if len(pareto_indices) == 0:
        return matrix_json(columns, [], [])

    indices = []
    data = []

    for epoch_idx in pareto_indices:
        epoch_idx = int(epoch_idx)
        record = post.history_[epoch_idx]

        indices.append(int(record.get("epoch")))

        data.append([
            float(value)
            for value in record.get("point", [])
        ])

    return matrix_json(columns, indices, data)


def pareto_unique_points_matrix(post):
    columns = post.metric_names_ or []

    if post.pareto_front_ is None or len(post.pareto_front_) == 0:
        return matrix_json(columns, [], [])

    pareto = np.asarray(post.pareto_front_, dtype=float)
    pareto_unique = np.unique(pareto, axis=0)

    indices = list(range(len(pareto_unique)))
    data = pareto_unique.tolist()

    return matrix_json(columns, indices, data)


def selected_solution(post):
    best_index = getattr(post, "best_index_", None)

    if best_index is None:
        return {}

    record = post.history_[best_index]

    thresholds = record.get("thresholds")

    if torch.is_tensor(thresholds):
        thresholds = thresholds.detach().cpu().tolist()

    return {
        "best_index": int(best_index),
        "epoch": int(record.get("epoch")),
        "metrics": to_jsonable(record.get("metrics")),
        "losses": to_jsonable(record.get("losses")),
        "point": to_jsonable(record.get("point")),
        "thresholds": to_jsonable(thresholds),
    }


# =========================================================
# Diagnóstico de gradientes
# =========================================================

def gradient_diagnostics_matrix(post):
    diagnostics = history_diagnostics(post)

    if len(diagnostics) == 0:
        return matrix_json([], [], [])

    columns = [
        "loss_raw_cross_entropy",
        "loss_raw_laplacian_fairness",

        "loss_norm_cross_entropy",
        "loss_norm_laplacian_fairness",

        "grad_norm_raw_cross_entropy",
        "grad_norm_raw_laplacian_fairness",

        "grad_norm_norm_cross_entropy",
        "grad_norm_norm_laplacian_fairness",

        "cosine_raw",
        "cosine_normalized",

        "alpha_cross_entropy",
        "alpha_laplacian_fairness",

        "dot_raw_cross_entropy",
        "dot_raw_laplacian_fairness",

        "dot_norm_cross_entropy",
        "dot_norm_laplacian_fairness",

        "descent_norm",
        "total_loss",
    ]

    indices = []
    data = []

    pair_name = "cross_entropy__laplacian_fairness"

    for row in diagnostics:
        losses_raw = row.get("losses_raw", {})
        losses_normalized = row.get("losses_normalized", {})

        grad_norms_raw = row.get("grad_norms_raw", {})
        grad_norms_normalized = row.get("grad_norms_normalized", {})

        cosine_raw = row.get("cosine_raw", {})
        cosine_normalized = row.get("cosine_normalized", {})

        descent_dot_raw = row.get("descent_dot_raw", {})
        descent_dot_normalized = row.get("descent_dot_normalized", {})

        alphas = row.get("alphas", [])

        indices.append(int(row["epoch"]))

        data.append([
            losses_raw.get("cross_entropy"),
            losses_raw.get("laplacian_fairness"),

            losses_normalized.get("cross_entropy"),
            losses_normalized.get("laplacian_fairness"),

            grad_norms_raw.get("cross_entropy"),
            grad_norms_raw.get("laplacian_fairness"),

            grad_norms_normalized.get("cross_entropy"),
            grad_norms_normalized.get("laplacian_fairness"),

            cosine_raw.get(pair_name),
            cosine_normalized.get(pair_name),

            alphas[0] if len(alphas) > 0 else None,
            alphas[1] if len(alphas) > 1 else None,

            descent_dot_raw.get("cross_entropy"),
            descent_dot_raw.get("laplacian_fairness"),

            descent_dot_normalized.get("cross_entropy"),
            descent_dot_normalized.get("laplacian_fairness"),

            row.get("descent_norm"),
            row.get("total_loss"),
        ])

    return matrix_json(columns, indices, data)


# =========================================================
# Utilidades para gráficos
# =========================================================

def transform_metric_for_plot(values, direction):
    values = np.asarray(values, dtype=float)

    if direction == "max":
        return values

    if direction == "min":
        return 1.0 - values

    raise ValueError(f"Direção desconhecida: {direction}")


def transformed_metric_name(name, direction):
    if direction == "max":
        return name

    if direction == "min":
        return f"1 - {name}"

    raise ValueError(f"Direção desconhecida: {direction}")


def get_matrix_column(matrix, column_name):
    columns = matrix["columns"]

    if column_name not in columns:
        return None

    j = columns.index(column_name)
    data = np.asarray(matrix["data"], dtype=float)

    if data.size == 0:
        return None

    return data[:, j]


# =========================================================
# Plots básicos
# =========================================================

def plot_losses(post, plots_dir):
    matrix = history_losses_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plt.figure()

    for j, name in enumerate(columns):
        plt.plot(indices, data[:, j], label=name)

    plt.title("Losses brutas por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "losses_por_epoca.png", dpi=150)
    plt.close()


def plot_metrics(post, plots_dir):
    matrix = history_metrics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plt.figure()

    for j, name in enumerate(columns):
        plt.plot(indices, data[:, j], label=name)

    plt.title("Métricas por época")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "metricas_por_epoca.png", dpi=150)
    plt.close()


def plot_thresholds(post, plots_dir):
    matrix = history_thresholds_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plt.figure()

    for j, name in enumerate(columns):
        plt.plot(indices, data[:, j], label=name)

    plt.title("Thresholds por época")
    plt.xlabel("Época")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "thresholds_por_epoca.png", dpi=150)
    plt.close()


def plot_pareto_2d_same_direction(post, plots_dir):
    metric_names = post.metric_names_ or []
    metric_directions = post.metric_directions_ or []

    if len(metric_names) != 2:
        return

    if len(metric_directions) != 2:
        return

    if len(post.history_) == 0:
        return

    all_points = np.asarray(
        [record["point"] for record in post.history_],
        dtype=float
    )

    pareto = np.asarray(post.pareto_front_, dtype=float)

    all_points_plot = all_points.copy()

    all_points_plot[:, 0] = transform_metric_for_plot(
        all_points[:, 0],
        metric_directions[0]
    )

    all_points_plot[:, 1] = transform_metric_for_plot(
        all_points[:, 1],
        metric_directions[1]
    )

    plt.figure()

    plt.scatter(
        all_points_plot[:, 0],
        all_points_plot[:, 1],
        label="Pontos por época",
        alpha=0.5
    )

    if pareto.size > 0:
        pareto_plot = pareto.copy()

        pareto_plot[:, 0] = transform_metric_for_plot(
            pareto[:, 0],
            metric_directions[0]
        )

        pareto_plot[:, 1] = transform_metric_for_plot(
            pareto[:, 1],
            metric_directions[1]
        )

        plt.scatter(
            pareto_plot[:, 0],
            pareto_plot[:, 1],
            label="Pareto",
            marker="x"
        )

    best_index = getattr(post, "best_index_", None)

    if best_index is not None:
        best_point = np.asarray(
            post.history_[best_index]["point"],
            dtype=float
        )

        best_point_plot = best_point.copy()

        best_point_plot[0] = transform_metric_for_plot(
            best_point[0],
            metric_directions[0]
        )

        best_point_plot[1] = transform_metric_for_plot(
            best_point[1],
            metric_directions[1]
        )

        plt.scatter(
            best_point_plot[0],
            best_point_plot[1],
            label="Selecionado",
            marker="*",
            s=150
        )

    xlabel = transformed_metric_name(
        metric_names[0],
        metric_directions[0]
    )

    ylabel = transformed_metric_name(
        metric_names[1],
        metric_directions[1]
    )

    plt.title("Fronteira de Pareto em escala de benefício")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_2d_same_direction.png", dpi=150)
    plt.close()


# =========================================================
# Plots de diagnóstico de gradientes
# =========================================================

def plot_normalized_losses(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plot_columns = [
        "loss_norm_cross_entropy",
        "loss_norm_laplacian_fairness",
    ]

    plt.figure()

    for name in plot_columns:
        if name not in columns:
            continue

        j = columns.index(name)
        plt.plot(indices, data[:, j], label=name)

    plt.title("Losses normalizadas por época")
    plt.xlabel("Época")
    plt.ylabel("Loss normalizada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "losses_normalizadas_por_epoca.png", dpi=150)
    plt.close()


def plot_alphas(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plot_columns = [
        "alpha_cross_entropy",
        "alpha_laplacian_fairness",
    ]

    plt.figure()

    for name in plot_columns:
        if name not in columns:
            continue

        j = columns.index(name)
        plt.plot(indices, data[:, j], label=name)

    plt.title("Alphas por época")
    plt.xlabel("Época")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "alphas_por_epoca.png", dpi=150)
    plt.close()


def plot_total_loss(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    if "total_loss" not in columns:
        return

    data = np.asarray(matrix["data"], dtype=float)
    j = columns.index("total_loss")

    plt.figure()
    plt.plot(indices, data[:, j])
    plt.title("Total loss por época")
    plt.xlabel("Época")
    plt.ylabel("Total loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "total_loss_por_epoca.png", dpi=150)
    plt.close()


def plot_gradient_norms(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plot_columns = [
        "grad_norm_raw_cross_entropy",
        "grad_norm_raw_laplacian_fairness",
        "grad_norm_norm_cross_entropy",
        "grad_norm_norm_laplacian_fairness",
    ]

    plt.figure()

    for name in plot_columns:
        if name not in columns:
            continue

        j = columns.index(name)
        plt.plot(indices, data[:, j], label=name)

    plt.title("Normas dos gradientes por época")
    plt.xlabel("Época")
    plt.ylabel("Norma")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "gradient_norms_por_epoca.png", dpi=150)
    plt.close()


def plot_cosines(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plot_columns = [
        "cosine_raw",
        "cosine_normalized",
    ]

    plt.figure()

    for name in plot_columns:
        if name not in columns:
            continue

        j = columns.index(name)
        plt.plot(indices, data[:, j], label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Similaridade cosseno entre objetivos")
    plt.xlabel("Época")
    plt.ylabel("Cosseno")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_por_epoca.png", dpi=150)
    plt.close()


def plot_descent_dot_products(post, plots_dir):
    matrix = gradient_diagnostics_matrix(post)

    columns = matrix["columns"]
    indices = matrix["indices"]

    if len(columns) == 0 or len(indices) == 0:
        return

    data = np.asarray(matrix["data"], dtype=float)

    plot_columns = [
        "dot_raw_cross_entropy",
        "dot_raw_laplacian_fairness",
        "dot_norm_cross_entropy",
        "dot_norm_laplacian_fairness",
    ]

    plt.figure()

    for name in plot_columns:
        if name not in columns:
            continue

        j = columns.index(name)
        plt.plot(indices, data[:, j], label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Produtos internos com a direção de descida")
    plt.xlabel("Época")
    plt.ylabel("Produto interno")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "descent_dot_products_por_epoca.png", dpi=150)
    plt.close()


# =========================================================
# Função principal
# =========================================================

def diagnose_postprocessor(post, output_dir):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    metadata_dir = output_dir / "metadata"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # CSVs matriciais
    # -----------------------------------------------------

    save_matrix_csv(
        metadata_dir / "points_by_epoch.csv",
        history_points_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "losses_by_epoch.csv",
        history_losses_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "metrics_by_epoch.csv",
        history_metrics_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "thresholds_by_epoch.csv",
        history_thresholds_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "pareto_points_with_epochs.csv",
        pareto_points_with_epochs_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "pareto_unique_points.csv",
        pareto_unique_points_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "gradient_diagnostics_by_epoch.csv",
        gradient_diagnostics_matrix(post)
    )

    # -----------------------------------------------------
    # JSONs estruturais
    # -----------------------------------------------------

    save_json(
        metadata_dir / "diagnostics_by_epoch.json",
        history_diagnostics(post)
    )

    save_json(
        metadata_dir / "selected_solution.json",
        selected_solution(post)
    )

    save_json(
        metadata_dir / "metadata.json",
        {
            "metric_names": post.metric_names_,
            "metric_directions": post.metric_directions_,
            "pareto_indices": getattr(post, "pareto_indices_", None),
            "best_index": getattr(post, "best_index_", None),
            "num_epochs": len(post.history_),
            "num_pareto_points": len(getattr(post, "pareto_front_", [])),
            "num_unique_pareto_points": len(
                pareto_unique_points_matrix(post)["data"]
            ),
            "has_gradient_diagnostics": len(history_diagnostics(post)) > 0,
        }
    )

    # -----------------------------------------------------
    # Gráficos
    # -----------------------------------------------------

    plot_losses(post, plots_dir)
    plot_normalized_losses(post, plots_dir)

    plot_metrics(post, plots_dir)
    plot_thresholds(post, plots_dir)
    plot_pareto_2d_same_direction(post, plots_dir)

    plot_total_loss(post, plots_dir)
    plot_alphas(post, plots_dir)

    plot_gradient_norms(post, plots_dir)
    plot_cosines(post, plots_dir)
    plot_descent_dot_products(post, plots_dir)

    return {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "metadata_dir": str(metadata_dir),
    }