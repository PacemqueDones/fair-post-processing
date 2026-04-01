import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def ddp(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)
    
    group_0 = (s == 0)
    group_1 = (s == 1)
    
    prob_1_g1 = y_pred[group_1].mean()
    prob_1_g0 = y_pred[group_0].mean()
    return float(abs(prob_1_g1 - prob_1_g0))

def deo(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)
    
    group_0 = (s == 0)
    group_1 = (s == 1)
    
    recall_g1 = y_pred[(group_1) & (y_true == 1)].mean()
    recall_g0 = y_pred[(group_0) & (y_true == 1)].mean()
    return float(abs(recall_g1 - recall_g0)) 

def calculate_metrics(y_true, y_pred, sensitive_features):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    diff_dp = ddp(y_true, y_pred, sensitive_features)
    diff_eo = deo(y_true, y_pred, sensitive_features)

    return {
        "acc": float(acc),
        "rec": float(rec),
        "prec": float(prec),
        "f1": float(f1),
        "ddp": float(diff_dp),
        "deo": float(diff_eo),
    }

def summarize_dict(title, d):
    print(title)
    for k, v in d.items():
        print(f"  {k}: {v:.6f}")
    print()


def get_metric_index(metric_names, name):
    try:
        return metric_names.index(name)
    except ValueError:
        return None


def print_best_points_on_pareto(post):
    pareto_all = np.asarray(post.pareto_front_, dtype=float)
    pareto_unique = np.unique(pareto_all, axis=0)

    metric_names = post.metric_names_
    directions = post.metric_directions_

    print("=== FRONTEIRA DE PARETO ===")
    print("Pareto total :", pareto_all.shape[0])
    print("Pareto única :", pareto_unique.shape[0])
    print("Razão de duplicação :", pareto_all.shape[0] / max(pareto_unique.shape[0], 1))
    print()

    print("Métricas usadas na seleção :", metric_names)
    print("Direções :", directions)
    print()

    for name, direction in zip(metric_names, directions):
        j = get_metric_index(metric_names, name)
        if j is None:
            continue

        if direction == "max":
            idx = np.argmax(pareto_unique[:, j])
        else:
            idx = np.argmin(pareto_unique[:, j])

        print(f"Melhor ponto em {name} ({direction}):")
        print(dict(zip(metric_names, pareto_unique[idx])))
        print()

    print("Ponto escolhido pelo selector:")
    print(post.best_metrics_)
    print()


def compare_selected_to_reference(post, y_val, s_val, model, X_val, y_test, s_test, preds, X_test):
    selected_test = calculate_metrics(y_test, preds, s_test)
    baseline_test = calculate_metrics(y_test, model.predict(X_test), s_test)
    baseline_val = calculate_metrics(y_val, model.predict(X_val), s_val)

    print("=== RESULTADO FINAL ===")
    print("Thresholds:", post.get_thresholds())
    print()

    summarize_dict("Solução com post-processing:", selected_test)
    summarize_dict("Solução sem post-processing:", baseline_test)
    summarize_dict("Referência na validação (modelo base):", baseline_val)

    if post.performance_reference_metrics_ is not None:
        print("Referência de desempenho armazenada no postprocessor:")
        print(post.performance_reference_metrics_)
        print()


def inspect_selector_gap(post):
    pareto_unique = np.unique(np.asarray(post.pareto_front_, dtype=float), axis=0)
    metric_names = post.metric_names_

    chosen = np.array([post.best_metrics_[name] for name in metric_names], dtype=float)

    print("=== DISTÂNCIA ENTRE ESCOLHIDO E EXTREMOS ===")
    for name in metric_names:
        j = metric_names.index(name)
        direction = post.metric_directions_[j]

        if direction == "max":
            best = pareto_unique[np.argmax(pareto_unique[:, j])]
        else:
            best = pareto_unique[np.argmin(pareto_unique[:, j])]

        gap = abs(best[j] - chosen[j])
        print(f"{name}: gap = {gap:.6f}")
    print()


def inspect_training_dynamics(post):
    print("=== DINÂMICA DO TREINO ===")

    if hasattr(post, "loss_history_") and len(post.loss_history_) > 0:
        keys = list(post.loss_history_[0].keys())
        print("Losses registradas:", keys)
    else:
        print("Sem loss_history_.")
    print()

    if hasattr(post, "total_loss_history_"):
        print("Total loss registrada:", len(post.total_loss_history_), "épocas")
    else:
        print("Sem total_loss_history_.")
    print()

    if hasattr(post, "alpha_history_"):
        print("Alpha history registrada:", len(post.alpha_history_), "épocas")
    elif hasattr(post, "alpha"):
        print("Alpha registrada em post.alpha:", len(post.alpha), "épocas")
    else:
        print("Sem alpha history.")
    print()


def plot_training_diagnostics(post):
    fig_count = 1

    # total loss
    if hasattr(post, "total_loss_history_") and len(post.total_loss_history_) > 0:
        plt.figure(fig_count)
        fig_count += 1
        plt.plot(post.total_loss_history_)
        plt.title("Total loss por época")
        plt.xlabel("Época")
        plt.ylabel("Total loss")
        plt.grid(True)

    # individual losses
    if hasattr(post, "loss_history_") and len(post.loss_history_) > 0:
        loss_keys = list(post.loss_history_[0].keys())
        plt.figure(fig_count)
        fig_count += 1
        for k in loss_keys:
            values = [d[k] for d in post.loss_history_]
            plt.plot(values, label=k)
        plt.title("Losses individuais por época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    # alphas
    alpha_hist = None
    if hasattr(post, "alpha_history_") and len(post.alpha_history_) > 0:
        alpha_hist = np.asarray(post.alpha_history_, dtype=float)
    elif hasattr(post, "alpha") and len(post.alpha) > 0:
        alpha_hist = np.asarray([
            a.detach().cpu().tolist() if hasattr(a, "detach") else list(a)
            for a in post.alpha
        ], dtype=float)

    if alpha_hist is not None and alpha_hist.ndim == 2:
        plt.figure(fig_count)
        fig_count += 1
        for j in range(alpha_hist.shape[1]):
            plt.plot(alpha_hist[:, j], label=f"alpha_{j}")
        plt.title("Alphas por época")
        plt.xlabel("Época")
        plt.ylabel("Alpha")
        plt.legend()
        plt.grid(True)

    # thresholds
    if hasattr(post, "threshold_history_") and len(post.threshold_history_) > 0:
        thr = np.asarray([
            t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            for t in post.threshold_history_
        ], dtype=float)

        if thr.ndim == 2:
            plt.figure(fig_count)
            fig_count += 1
            for j in range(thr.shape[1]):
                plt.plot(thr[:, j], label=f"threshold_{j}")
            plt.title("Thresholds por época")
            plt.xlabel("Época")
            plt.ylabel("Threshold")
            plt.legend()
            plt.grid(True)

    plt.show()


def diagnose_postprocessor(post, model, X_val, y_val, s_val, X_test, y_test, s_test, preds):
    compare_selected_to_reference(post, y_val, s_val, model, X_val, y_test, s_test, preds, X_test)
    print_best_points_on_pareto(post)
    inspect_selector_gap(post)
    inspect_training_dynamics(post)

    print("=== FILTRO DE PERFORMANCE ===")
    print("fallback_used :", getattr(post, "fallback_used_", None))
    print("fallback_reason :", getattr(post, "fallback_reason_", None))
    print("filter_info :", getattr(post, "filter_info_", None))
    print()

    if getattr(post, "feasible_pareto_front_", None) is not None:
        feasible = np.asarray(post.feasible_pareto_front_, dtype=float)
        feasible_unique = np.unique(feasible, axis=0)
        print("Feasible total :", feasible.shape[0])
        print("Feasible única :", feasible_unique.shape[0])
        print()

    plot_training_diagnostics(post)