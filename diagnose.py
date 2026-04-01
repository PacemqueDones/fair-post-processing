import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# =========================================================
# MÉTRICAS
# =========================================================

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


# =========================================================
# UTILITÁRIOS
# =========================================================

def summarize_dict(title, d):
    print(title)
    for k, v in d.items():
        print(f"  {k}: {v:.6f}")
    print()


def safe_mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if len(x) > 0 else None


def safe_std(x):
    x = [v for v in x if v is not None]
    return float(np.std(x)) if len(x) > 0 else None


def get_metric_index(metric_names, name):
    try:
        return metric_names.index(name)
    except ValueError:
        return None


def get_selected_epoch(post):
    """
    Tenta localizar a época correspondente ao best_thresholds_.
    """
    if not hasattr(post, "threshold_history_") or post.best_thresholds_ is None:
        return None

    best = post.best_thresholds_.detach().cpu().numpy()

    for i, thr in enumerate(post.threshold_history_):
        arr = thr.detach().cpu().numpy()
        if np.allclose(arr, best, atol=1e-8):
            return i

    return None


# =========================================================
# RESULTADO FINAL
# =========================================================

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


# =========================================================
# FRONTEIRA DE PARETO
# =========================================================

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


def inspect_selected_epoch(post):
    epoch = get_selected_epoch(post)
    print("=== ÉPOCA ESCOLHIDA ===")
    print("Época correspondente ao best_thresholds_:", epoch)

    if epoch is not None:
        if hasattr(post, "loss_history_") and epoch < len(post.loss_history_):
            print("Losses da época escolhida:")
            print(post.loss_history_[epoch])
        if hasattr(post, "metric_history_") and epoch < len(post.metric_history_):
            print("Métricas da época escolhida:")
            print(post.metric_history_[epoch])
        if hasattr(post, "alpha_history_") and epoch < len(post.alpha_history_):
            print("Alphas da época escolhida:")
            print(post.alpha_history_[epoch])
        if hasattr(post, "grad_norm_history_") and epoch < len(post.grad_norm_history_):
            print("Normas dos gradientes da época escolhida:")
            print(post.grad_norm_history_[epoch])
        if hasattr(post, "cosine_similarity_history_") and epoch < len(post.cosine_similarity_history_):
            print("Similaridade cosseno da época escolhida:")
            print(post.cosine_similarity_history_[epoch])
    print()


# =========================================================
# DINÂMICA DO TREINO
# =========================================================

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
    else:
        print("Sem alpha history.")
    print()

    if hasattr(post, "grad_norm_history_"):
        print("Grad norm history registrada:", len(post.grad_norm_history_), "épocas")
    else:
        print("Sem grad_norm_history_.")
    print()

    if hasattr(post, "cosine_similarity_history_"):
        print("Cosine similarity history registrada:", len(post.cosine_similarity_history_), "épocas")
    else:
        print("Sem cosine_similarity_history_.")
    print()

    if hasattr(post, "step_norm_history_"):
        print("Step norm history registrada:", len(post.step_norm_history_), "épocas")
    else:
        print("Sem step_norm_history_.")
    print()


def summarize_training_statistics(post):
    print("=== RESUMO ESTATÍSTICO DO TREINO ===")

    # total loss
    if hasattr(post, "total_loss_history_") and len(post.total_loss_history_) > 0:
        tl = np.asarray(post.total_loss_history_, dtype=float)
        print(f"Total loss: início={tl[0]:.6f}, fim={tl[-1]:.6f}, delta={tl[-1]-tl[0]:.6f}")
    print()

    # losses individuais
    if hasattr(post, "loss_history_") and len(post.loss_history_) > 0:
        keys = list(post.loss_history_[0].keys())
        for k in keys:
            vals = np.asarray([d[k] for d in post.loss_history_], dtype=float)
            print(f"{k}: início={vals[0]:.6f}, fim={vals[-1]:.6f}, delta={vals[-1]-vals[0]:.6f}")
    print()

    # alphas
    if hasattr(post, "alpha_history_") and len(post.alpha_history_) > 0:
        alpha_hist = np.asarray(post.alpha_history_, dtype=float)
        if alpha_hist.ndim == 2:
            for j in range(alpha_hist.shape[1]):
                vals = alpha_hist[:, j]
                print(
                    f"alpha_{j}: média={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                    f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
                )
    print()

    # grad norms
    if hasattr(post, "grad_norm_history_") and len(post.grad_norm_history_) > 0:
        grad_keys = list(post.grad_norm_history_[0].keys())
        for k in grad_keys:
            vals = np.asarray([d[k] for d in post.grad_norm_history_], dtype=float)
            print(
                f"||grad {k}||: média={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
            )
    print()

    # cosine similarity
    if hasattr(post, "cosine_similarity_history_") and len(post.cosine_similarity_history_) > 0:
        first = post.cosine_similarity_history_[0]
        if isinstance(first, dict) and len(first) > 0:
            for key in first.keys():
                vals = [d.get(key) for d in post.cosine_similarity_history_]
                vals = [v for v in vals if v is not None]
                if len(vals) > 0:
                    vals = np.asarray(vals, dtype=float)
                    print(
                        f"{key}: cosine média={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
                        f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
                    )
    print()

    # step norm
    if hasattr(post, "step_norm_history_") and len(post.step_norm_history_) > 0:
        vals = np.asarray(post.step_norm_history_, dtype=float)
        print(
            f"||Δthresholds||: média={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
            f"min={np.min(vals):.6f}, max={np.max(vals):.6f}"
        )
    print()


def automatic_diagnosis(post):
    print("=== DIAGNÓSTICO AUTOMÁTICO ===")

    # 1. dominância dos alphas
    if hasattr(post, "alpha_history_") and len(post.alpha_history_) > 0:
        alpha_hist = np.asarray(post.alpha_history_, dtype=float)
        if alpha_hist.ndim == 2 and alpha_hist.shape[1] == 2:
            a0 = alpha_hist[:, 0]
            a1 = alpha_hist[:, 1]

            if np.allclose(a0, 1.0) and np.allclose(a1, 0.0):
                print("- O treino foi dominado totalmente pelo objetivo 0.")
            elif np.allclose(a1, 1.0) and np.allclose(a0, 0.0):
                print("- O treino foi dominado totalmente pelo objetivo 1.")
            elif np.mean(a0) > 0.9:
                print("- O objetivo 0 dominou fortemente o treino.")
            elif np.mean(a1) > 0.9:
                print("- O objetivo 1 dominou fortemente o treino.")
            else:
                print("- Houve participação mista dos dois objetivos.")
    print()

    # 2. gradientes
    if hasattr(post, "grad_norm_history_") and len(post.grad_norm_history_) > 0:
        keys = list(post.grad_norm_history_[0].keys())
        if len(keys) == 2:
            k0, k1 = keys[0], keys[1]
            g0 = np.mean([d[k0] for d in post.grad_norm_history_])
            g1 = np.mean([d[k1] for d in post.grad_norm_history_])

            ratio = max(g0, g1) / max(min(g0, g1), 1e-12)

            print(f"- Razão média entre normas dos gradientes: {ratio:.4f}")
            if ratio > 10:
                print("  -> Há forte desbalanceamento de escala entre os gradientes.")
            elif ratio > 3:
                print("  -> Há desbalanceamento moderado entre os gradientes.")
            else:
                print("  -> As normas dos gradientes estão em escala relativamente próxima.")
    print()

    # 3. cosine similarity
    if hasattr(post, "cosine_similarity_history_") and len(post.cosine_similarity_history_) > 0:
        first = post.cosine_similarity_history_[0]
        if isinstance(first, dict) and len(first) > 0:
            key = list(first.keys())[0]
            vals = [d.get(key) for d in post.cosine_similarity_history_ if d.get(key) is not None]
            if len(vals) > 0:
                mean_cos = float(np.mean(vals))
                print(f"- Similaridade cosseno média entre gradientes: {mean_cos:.6f}")
                if mean_cos > 0.9:
                    print("  -> Os objetivos estão quase alinhados.")
                elif mean_cos > 0.3:
                    print("  -> Os objetivos têm alinhamento parcial.")
                elif mean_cos > -0.3:
                    print("  -> Os objetivos estão pouco correlacionados.")
                else:
                    print("  -> Os objetivos estão em conflito significativo.")
    print()

    # 4. step norm
    if hasattr(post, "step_norm_history_") and len(post.step_norm_history_) > 0:
        mean_step = float(np.mean(post.step_norm_history_))
        print(f"- Norma média do passo dos thresholds: {mean_step:.8f}")
        if mean_step < 1e-6:
            print("  -> Os thresholds quase não estão se movendo.")
        elif mean_step < 1e-4:
            print("  -> O movimento dos thresholds é pequeno.")
        else:
            print("  -> Os thresholds estão se movendo de forma perceptível.")
    print()

    # 5. duplicação da fronteira
    pareto_all = np.asarray(post.pareto_front_, dtype=float)
    pareto_unique = np.unique(pareto_all, axis=0)
    duplication_ratio = pareto_all.shape[0] / max(pareto_unique.shape[0], 1)

    print(f"- Razão de duplicação da fronteira: {duplication_ratio:.4f}")
    if duplication_ratio > 10:
        print("  -> A fronteira está muito redundante.")
    elif duplication_ratio > 3:
        print("  -> A fronteira tem redundância moderada.")
    else:
        print("  -> A fronteira tem boa diversidade.")
    print()


# =========================================================
# PLOTS
# =========================================================

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

    # losses individuais
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

    # deltas das losses
    if hasattr(post, "loss_delta_history_") and len(post.loss_delta_history_) > 0:
        delta_keys = list(post.loss_delta_history_[0].keys())
        plt.figure(fig_count)
        fig_count += 1
        for k in delta_keys:
            values = [d[k] for d in post.loss_delta_history_]
            plt.plot(values, label=f"delta_{k}")
        plt.title("Delta das losses por época")
        plt.xlabel("Época")
        plt.ylabel("Delta")
        plt.legend()
        plt.grid(True)

    # alphas
    if hasattr(post, "alpha_history_") and len(post.alpha_history_) > 0:
        alpha_hist = np.asarray(post.alpha_history_, dtype=float)
        if alpha_hist.ndim == 2:
            plt.figure(fig_count)
            fig_count += 1
            for j in range(alpha_hist.shape[1]):
                plt.plot(alpha_hist[:, j], label=f"alpha_{j}")
            plt.title("Alphas por época")
            plt.xlabel("Época")
            plt.ylabel("Alpha")
            plt.legend()
            plt.grid(True)

    # grad norms
    if hasattr(post, "grad_norm_history_") and len(post.grad_norm_history_) > 0:
        grad_keys = list(post.grad_norm_history_[0].keys())
        plt.figure(fig_count)
        fig_count += 1
        for k in grad_keys:
            values = [d[k] for d in post.grad_norm_history_]
            plt.plot(values, label=f"grad_{k}")
        plt.title("Norma dos gradientes por época")
        plt.xlabel("Época")
        plt.ylabel("||grad||")
        plt.legend()
        plt.grid(True)

    # cosine similarity
    if hasattr(post, "cosine_similarity_history_") and len(post.cosine_similarity_history_) > 0:
        first = post.cosine_similarity_history_[0]
        if isinstance(first, dict) and len(first) > 0:
            for key in first.keys():
                vals = [d.get(key) for d in post.cosine_similarity_history_]
                vals = [v for v in vals if v is not None]
                if len(vals) > 0:
                    plt.figure(fig_count)
                    fig_count += 1
                    plt.plot(vals)
                    plt.title(f"Similaridade cosseno: {key}")
                    plt.xlabel("Época")
                    plt.ylabel("Cosine similarity")
                    plt.grid(True)

    # step norm
    if hasattr(post, "step_norm_history_") and len(post.step_norm_history_) > 0:
        plt.figure(fig_count)
        fig_count += 1
        plt.plot(post.step_norm_history_)
        plt.title("Norma do passo por época")
        plt.xlabel("Época")
        plt.ylabel("||Δthresholds||")
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


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def diagnose_postprocessor(post, model, X_val, y_val, s_val, X_test, y_test, s_test, preds):
    compare_selected_to_reference(post, y_val, s_val, model, X_val, y_test, s_test, preds, X_test)
    print_best_points_on_pareto(post)
    inspect_selector_gap(post)
    inspect_selected_epoch(post)
    inspect_training_dynamics(post)
    summarize_training_statistics(post)
    automatic_diagnosis(post)

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