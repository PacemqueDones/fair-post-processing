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
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

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
            values = [str(idx)] + [str(value) for value in row]
            f.write(",".join(values) + "\n")

def matrix_json(columns, indices, data):
    return {
        "columns": columns,
        "indices": indices,
        "data": data,
    }


# =========================================================
# Extração matricial do histórico
# =========================================================

def history_points_matrix(post):
    columns = post.metric_names_ or []

    indices = []
    data = []

    for record in post.history_:
        point = record.get("point", [])

        indices.append(int(record.get("epoch")))

        data.append([
            float(value)
            for value in point
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
# Pareto em formato matricial
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
        point = record.get("point", [])

        indices.append(int(record.get("epoch")))

        data.append([
            float(value)
            for value in point
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
# Similaridade cosseno
# =========================================================

def get_objective_names(post):
    if len(post.history_) == 0:
        return []

    return list(post.history_[0].get("losses", {}).keys())


def parse_cosine_key(key, objective_names):
    if isinstance(key, tuple) and len(key) == 2:
        return key[0], key[1]

    key = str(key)

    separators = [
        " vs ",
        "_vs_",
        "__",
        "--",
        " - ",
        ",",
        "|",
    ]

    for sep in separators:
        if sep in key:
            left, right = key.split(sep, 1)
            return left.strip(), right.strip()

    if len(objective_names) == 2:
        return objective_names[0], objective_names[1]

    return None


def cosine_similarity_by_epoch_matrix(post):
    objective_names = get_objective_names(post)

    if len(objective_names) < 2:
        return matrix_json([], [], [])

    columns = []
    pair_names = []

    for i in range(len(objective_names)):
        for j in range(i + 1, len(objective_names)):
            pair_name = f"{objective_names[i]}__{objective_names[j]}"
            columns.append(pair_name)
            pair_names.append((objective_names[i], objective_names[j]))

    indices = []
    data = []

    for record in post.history_:
        diagnostics = record.get("diagnostics")

        if diagnostics is None:
            continue

        cosine_dict = diagnostics.get("cosine_similarity", {})

        if cosine_dict is None or len(cosine_dict) == 0:
            continue

        row = []

        for obj_i, obj_j in pair_names:
            value_found = None

            for key, value in cosine_dict.items():
                parsed = parse_cosine_key(key, objective_names)

                if parsed is None:
                    continue

                left, right = parsed

                same_order = left == obj_i and right == obj_j
                reverse_order = left == obj_j and right == obj_i

                if same_order or reverse_order:
                    value_found = float(value)
                    break

            row.append(value_found)

        indices.append(int(record.get("epoch")))
        data.append(row)

    return matrix_json(columns, indices, data)


def cosine_similarity_matrix(post):
    objective_names = get_objective_names(post)
    n = len(objective_names)

    if n == 0:
        return matrix_json([], [], [])

    matrix_sum = np.eye(n, dtype=float)
    matrix_count = np.eye(n, dtype=float)

    name_to_idx = {
        name: i
        for i, name in enumerate(objective_names)
    }

    for record in post.history_:
        diagnostics = record.get("diagnostics")

        if diagnostics is None:
            continue

        cosine_dict = diagnostics.get("cosine_similarity", {})

        if cosine_dict is None:
            continue

        for key, value in cosine_dict.items():
            parsed = parse_cosine_key(key, objective_names)

            if parsed is None:
                continue

            obj_i, obj_j = parsed

            if obj_i not in name_to_idx or obj_j not in name_to_idx:
                continue

            i = name_to_idx[obj_i]
            j = name_to_idx[obj_j]

            matrix_sum[i, j] += float(value)
            matrix_sum[j, i] += float(value)

            matrix_count[i, j] += 1
            matrix_count[j, i] += 1

    matrix = matrix_sum / np.maximum(matrix_count, 1)

    return matrix_json(
        columns=objective_names,
        indices=objective_names,
        data=matrix.tolist()
    )


# =========================================================
# Plots
# =========================================================

def plot_losses(post, plots_dir):
    if len(post.history_) == 0:
        return

    first_losses = post.history_[0].get("losses", {})

    if len(first_losses) == 0:
        return

    epochs = [record["epoch"] for record in post.history_]

    plt.figure()

    for loss_name in first_losses.keys():
        values = [
            record["losses"].get(loss_name)
            for record in post.history_
        ]

        plt.plot(epochs, values, label=loss_name)

    plt.title("Losses por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "losses_por_epoca.png", dpi=150)
    plt.close()


def plot_metrics(post, plots_dir):
    if len(post.history_) == 0:
        return

    first_metrics = post.history_[0].get("metrics", {})

    if len(first_metrics) == 0:
        return

    epochs = [record["epoch"] for record in post.history_]

    plt.figure()

    for metric_name in first_metrics.keys():
        values = [
            record["metrics"].get(metric_name)
            for record in post.history_
        ]

        plt.plot(epochs, values, label=metric_name)

    plt.title("Métricas por época")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "metricas_por_epoca.png", dpi=150)
    plt.close()


def plot_thresholds(post, plots_dir):
    if len(post.history_) == 0:
        return

    thresholds = []

    for record in post.history_:
        thr = record.get("thresholds")

        if thr is None:
            return

        if torch.is_tensor(thr):
            thr = thr.detach().cpu().numpy()
        else:
            thr = np.asarray(thr)

        thresholds.append(thr)

    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 2:
        return

    epochs = [record["epoch"] for record in post.history_]

    plt.figure()

    for j in range(thresholds.shape[1]):
        plt.plot(epochs, thresholds[:, j], label=f"threshold_{j}")

    plt.title("Thresholds por época")
    plt.xlabel("Época")
    plt.ylabel("Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "thresholds_por_epoca.png", dpi=150)
    plt.close()


def plot_pareto_2d(post, plots_dir):
    metric_names = post.metric_names_ or []

    if len(metric_names) != 2:
        return

    if len(post.history_) == 0:
        return

    all_points = np.asarray(
        [record["point"] for record in post.history_],
        dtype=float
    )

    pareto = np.asarray(post.pareto_front_, dtype=float)

    plt.figure()

    plt.scatter(
        all_points[:, 0],
        all_points[:, 1],
        label="Pontos por época",
        alpha=0.5
    )

    if pareto.size > 0:
        plt.scatter(
            pareto[:, 0],
            pareto[:, 1],
            label="Pareto",
            marker="x"
        )

    best_index = getattr(post, "best_index_", None)

    if best_index is not None:
        best_point = np.asarray(post.history_[best_index]["point"], dtype=float)

        plt.scatter(
            best_point[0],
            best_point[1],
            label="Selecionado",
            marker="*",
            s=150
        )

    plt.title("Pontos e fronteira de Pareto")
    plt.xlabel(metric_names[0])
    plt.ylabel(metric_names[1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_2d.png", dpi=150)
    plt.close()


def plot_alphas(post, plots_dir):
    diagnostics = history_diagnostics(post)

    if len(diagnostics) == 0:
        return

    if "alphas" not in diagnostics[0]:
        return

    epochs = [row["epoch"] for row in diagnostics]
    alphas = np.asarray([row["alphas"] for row in diagnostics], dtype=float)

    if alphas.ndim != 2:
        return

    plt.figure()

    for j in range(alphas.shape[1]):
        plt.plot(epochs, alphas[:, j], label=f"alpha_{j}")

    plt.title("Alphas por época")
    plt.xlabel("Época")
    plt.ylabel("Alpha")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "alphas_por_epoca.png", dpi=150)
    plt.close()


def plot_total_loss(post, plots_dir):
    diagnostics = history_diagnostics(post)

    if len(diagnostics) == 0:
        return

    if "total_loss" not in diagnostics[0]:
        return

    epochs = [row["epoch"] for row in diagnostics]
    values = [row["total_loss"] for row in diagnostics]

    plt.figure()
    plt.plot(epochs, values)
    plt.title("Total loss por época")
    plt.xlabel("Época")
    plt.ylabel("Total loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "total_loss_por_epoca.png", dpi=150)
    plt.close()


def plot_cosine_similarity_lines(post, plots_dir):
    cosine = cosine_similarity_by_epoch_matrix(post)

    columns = cosine["columns"]
    indices = cosine["indices"]
    data = np.asarray(cosine["data"], dtype=float)

    if len(columns) == 0 or len(indices) == 0:
        return

    plt.figure()

    for j, name in enumerate(columns):
        plt.plot(indices, data[:, j], label=name)

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.title("Similaridade cosseno entre objetivos por época")
    plt.xlabel("Época")
    plt.ylabel("Similaridade cosseno")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_por_epoca.png", dpi=150)
    plt.close()


def plot_cosine_similarity_matrix(post, plots_dir):
    result = cosine_similarity_matrix(post)

    objective_names = result["columns"]
    matrix = np.asarray(result["data"], dtype=float)

    if len(objective_names) == 0:
        return

    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        matrix,
        vmin=-1,
        vmax=1
    )

    plt.colorbar(im, label="Similaridade cosseno média")

    plt.xticks(
        ticks=np.arange(len(objective_names)),
        labels=objective_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        ticks=np.arange(len(objective_names)),
        labels=objective_names
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center"
            )

    plt.title("Matriz média de similaridade cosseno entre objetivos")
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_matrix.png", dpi=150)
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
    # JSONs principais em formato matricial
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
        metadata_dir / "cosine_similarity_by_epoch.csv",
        cosine_similarity_by_epoch_matrix(post)
    )

    save_matrix_csv(
        metadata_dir / "cosine_similarity_matrix.csv",
        cosine_similarity_matrix(post)
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
            "has_cosine_similarity": len(
                cosine_similarity_by_epoch_matrix(post)["data"]
            ) > 0,
        }
    )

    # -----------------------------------------------------
    # Gráficos
    # -----------------------------------------------------

    plot_losses(post, plots_dir)
    plot_metrics(post, plots_dir)
    plot_thresholds(post, plots_dir)
    plot_pareto_2d(post, plots_dir)
    plot_total_loss(post, plots_dir)
    plot_alphas(post, plots_dir)
    plot_cosine_similarity_lines(post, plots_dir)
    plot_cosine_similarity_matrix(post, plots_dir)

    return {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "metadata_dir": str(metadata_dir),
    }