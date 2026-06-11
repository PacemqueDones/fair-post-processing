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


# =========================================================
# Extração do histórico
# =========================================================

def history_points(post):
    rows = []

    metric_names = post.metric_names_ or []

    for record in post.history_:
        point = record.get("point", [])

        row = {
            "epoch": record.get("epoch"),
        }

        for name, value in zip(metric_names, point):
            row[name] = float(value)

        row["point"] = point

        rows.append(row)

    return rows


def history_losses(post):
    rows = []

    for record in post.history_:
        row = {
            "epoch": record.get("epoch"),
        }

        losses = record.get("losses", {})
        for name, value in losses.items():
            row[name] = float(value)

        rows.append(row)

    return rows


def history_metrics(post):
    rows = []

    for record in post.history_:
        row = {
            "epoch": record.get("epoch"),
        }

        metrics = record.get("metrics", {})
        for name, value in metrics.items():
            row[name] = float(value)

        rows.append(row)

    return rows


def history_thresholds(post):
    rows = []

    for record in post.history_:
        thresholds = record.get("thresholds")

        row = {
            "epoch": record.get("epoch"),
            "thresholds": to_jsonable(thresholds),
        }

        rows.append(row)

    return rows


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

def pareto_unique_points(post):
    metric_names = post.metric_names_ or []

    if post.pareto_front_ is None or len(post.pareto_front_) == 0:
        return []

    pareto = np.asarray(post.pareto_front_, dtype=float)
    pareto_unique = np.unique(pareto, axis=0)

    rows = []

    for point in pareto_unique:
        row = {}

        for name, value in zip(metric_names, point):
            row[name] = float(value)

        row["point"] = point.tolist()

        rows.append(row)

    return rows


def pareto_points_with_epochs(post):
    rows = []

    metric_names = post.metric_names_ or []
    pareto_indices = getattr(post, "pareto_indices_", None)

    if pareto_indices is None:
        return rows

    if torch.is_tensor(pareto_indices):
        pareto_indices = pareto_indices.detach().cpu().tolist()

    elif isinstance(pareto_indices, np.ndarray):
        pareto_indices = pareto_indices.tolist()

    for epoch_idx in pareto_indices:
        epoch_idx = int(epoch_idx)

        record = post.history_[epoch_idx]
        point = record.get("point", [])

        row = {
            "epoch": record.get("epoch"),
        }

        for name, value in zip(metric_names, point):
            row[name] = float(value)

        row["point"] = point

        rows.append(row)

    return rows

def selected_solution(post):
    best_index = getattr(post, "best_index_", None)

    if best_index is None:
        return {}

    record = post.history_[best_index]

    return {
        "best_index": best_index,
        "epoch": record.get("epoch"),
        "metrics": record.get("metrics"),
        "losses": record.get("losses"),
        "point": record.get("point"),
        "thresholds": record.get("thresholds"),
    }


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


# =========================================================
# Função principal
# =========================================================

def diagnose_postprocessor(post, output_dir):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    json_dir = output_dir / "json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # JSONs principais
    # -----------------------------------------------------

    save_json(
        json_dir / "points_by_epoch.json",
        history_points(post)
    )

    save_json(
        json_dir / "losses_by_epoch.json",
        history_losses(post)
    )

    save_json(
        json_dir / "metrics_by_epoch.json",
        history_metrics(post)
    )

    save_json(
        json_dir / "thresholds_by_epoch.json",
        history_thresholds(post)
    )

    save_json(
        json_dir / "diagnostics_by_epoch.json",
        history_diagnostics(post)
    )

    save_json(
        json_dir / "pareto_points_with_epochs.json",
        pareto_points_with_epochs(post)
    )

    save_json(
        json_dir / "pareto_unique_points.json",
        pareto_unique_points(post)
    )

    save_json(
        json_dir / "selected_solution.json",
        selected_solution(post)
    )

    save_json(
        json_dir / "metadata.json",
        {
            "metric_names": post.metric_names_,
            "metric_directions": post.metric_directions_,
            "pareto_indices": getattr(post, "pareto_indices_", None),
            "best_index": getattr(post, "best_index_", None),
            "num_epochs": len(post.history_),
            "num_pareto_points": len(getattr(post, "pareto_front_", [])),
            "num_unique_pareto_points": len(pareto_unique_points(post)),
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

    return {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "json_dir": str(json_dir),
    }