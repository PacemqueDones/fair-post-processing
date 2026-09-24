import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def to_jsonable(value):
    """Converte valores usados pelo FairPP em valores serializáveis."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def history_matrix(post, field, columns=None):
    """Organiza um campo do history_ em uma matriz indexada pela época."""
    if columns is None:
        columns = []
        for record in post.history_:
            for name in record.get(field, {}):
                if name not in columns:
                    columns.append(name)

    epochs = []
    rows = []
    for record in post.history_:
        values = record.get(field, {})
        epochs.append(int(record["epoch"]))
        rows.append([
            None if name not in values else float(values[name])
            for name in columns
        ])

    return {"columns": columns, "epochs": epochs, "rows": rows}


def history_metrics_matrix(post):
    return history_matrix(post, "metrics")


def history_losses_matrix(post):
    return history_matrix(post, "losses")


def pareto_unique_points_matrix(post):
    if post.pareto_front_:
        points = np.unique(np.asarray(post.pareto_front_, dtype=float), axis=0)
    else:
        points = np.empty((0, len(post.metric_names_ or [])))

    return {
        "columns": post.metric_names_ or [],
        "epochs": list(range(len(points))),
        "rows": points.tolist(),
    }


def _component_configuration(component):
    parameters = {}
    for name, value in vars(component).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            parameters[name] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool))
            for item in value
        ):
            parameters[name] = list(value)

    return {"class": component.__class__.__name__, "parameters": parameters}


def training_configuration(post, run_config):
    configuration = {
        "postprocessor": {
            "aggregator": post.aggregator_name,
            "learning_rate": post.lr,
            "epochs": post.epochs,
        },
        "model": _component_configuration(post.model),
        "objectives": [
            _component_configuration(objective)
            for objective in post.objectives
        ],
        "selector": _component_configuration(post.selector),
        "selection_metrics": [
            {
                "name": metric.name,
                "direction": metric.direction,
                "type": metric.type,
            }
            for metric in post.selection_metrics
        ],
    }
    if run_config is not None:
        configuration["experiment"] = run_config
    return configuration


def selected_solution(post):
    record = post.history_[post.best_index_]
    return {
        "selected_history_index": int(post.best_index_),
        "selected_epoch": int(record["epoch"]),
        "metrics": record["metrics"],
        "losses": record["losses"],
        "point": record["point"],
        "num_recorded_epochs": len(post.history_),
        "num_pareto_points": len(post.pareto_front_),
        "num_unique_pareto_points": len(pareto_unique_points_matrix(post)["rows"]),
    }


def _plot_history(matrix, title, ylabel, path):
    if not matrix["columns"]:
        return

    epochs = np.asarray(matrix["epochs"])
    values = np.asarray(matrix["rows"], dtype=float)
    plt.figure()
    for column_index, name in enumerate(matrix["columns"]):
        plt.plot(epochs, values[:, column_index], label=name)

    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_losses(post, plots_dir):
    _plot_history(
        history_losses_matrix(post),
        title="Losses de treino por época",
        ylabel="Loss",
        path=plots_dir / "losses_por_epoca.png",
    )


def plot_metrics(post, plots_dir):
    _plot_history(
        history_metrics_matrix(post),
        title="Métricas de validação por época",
        ylabel="Valor",
        path=plots_dir / "metricas_por_epoca.png",
    )


def _benefit_scale(values, direction):
    values = np.asarray(values, dtype=float)
    if direction == "max":
        return values
    if direction == "min":
        return 1.0 - values
    raise ValueError(f"Direção desconhecida: {direction}")


def _benefit_label(name, direction):
    return name if direction == "max" else f"1 - {name}"


def plot_pareto_2d(post, plots_dir):
    if len(post.metric_names_ or []) != 2:
        return
    if len(post.metric_directions_ or []) != 2:
        return

    points = np.asarray([record["point"] for record in post.history_], dtype=float)
    pareto_points = np.asarray(post.pareto_front_, dtype=float)
    points[:, 0] = _benefit_scale(points[:, 0], post.metric_directions_[0])
    points[:, 1] = _benefit_scale(points[:, 1], post.metric_directions_[1])

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label="Épocas")
    plt.scatter(points[0, 0], points[0, 1], marker="o", s=70, label="Inicial")

    if pareto_points.size:
        pareto_points[:, 0] = _benefit_scale(
            pareto_points[:, 0], post.metric_directions_[0]
        )
        pareto_points[:, 1] = _benefit_scale(
            pareto_points[:, 1], post.metric_directions_[1]
        )
        plt.scatter(
            pareto_points[:, 0], pareto_points[:, 1], marker="x", label="Pareto"
        )

    selected_point = points[post.best_index_]
    plt.scatter(
        selected_point[0], selected_point[1], marker="*", s=150, label="Selecionado"
    )
    plt.title("Fronteira de Pareto em escala de benefício")
    plt.xlabel(_benefit_label(post.metric_names_[0], post.metric_directions_[0]))
    plt.ylabel(_benefit_label(post.metric_names_[1], post.metric_directions_[1]))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "pareto_2d.png", dpi=150)
    plt.close()


def diagnose_postprocessor(post, output_dir, run_config=None):
    """Salva o painel de diagnóstico de um pós-processamento treinado."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "run_config.json",
        training_configuration(post, run_config),
    )
    save_json(
        output_dir / "training_summary.json",
        selected_solution(post),
    )

    plot_losses(post, output_dir)
    plot_metrics(post, output_dir)
    plot_pareto_2d(post, output_dir)

    return {"output_dir": str(output_dir)}
