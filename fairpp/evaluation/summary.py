import numpy as np


def resumir_resultados(lista_resultados):
    keys = lista_resultados[0].keys()
    resumo = {}

    for key in keys:
        valores = np.array([r[key] for r in lista_resultados])

        resumo[key] = {
            "mean": float(valores.mean()),
            "std": float(valores.std(ddof=1))
        }

    return resumo