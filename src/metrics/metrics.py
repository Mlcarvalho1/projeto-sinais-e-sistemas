"""Métricas objetivas de denoising de ECG (módulo compartilhado com o time).

Mesmas métricas da Entrega 2: SNR (dB) de entrada/saída e melhoria, RMSE, PRD
(percentage root-mean-square difference) e coeficiente de correlação de Pearson.
Todas comparam um sinal contra o ECG de referência ("limpo").
"""
from __future__ import annotations

import numpy as np


def _align(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def snr_db(clean, x) -> float:
    """SNR de ``x`` vs. ``clean``: ``10*log10(P_sinal / P_ruído)``, ruído = ``x - clean``."""
    clean, x = _align(clean, x)
    p_sig = np.sum(clean ** 2)
    p_noise = np.sum((x - clean) ** 2)
    return float(10 * np.log10(p_sig / p_noise))


def snr_improvement(clean, noisy, filtered) -> float:
    """Melhoria de SNR (dB): SNR(filtrado) - SNR(ruidoso)."""
    return snr_db(clean, filtered) - snr_db(clean, noisy)


def rmse(clean, x) -> float:
    clean, x = _align(clean, x)
    return float(np.sqrt(np.mean((clean - x) ** 2)))


def prd(clean, x) -> float:
    """Percentage Root-mean-square Difference (%)."""
    clean, x = _align(clean, x)
    return float(100 * np.sqrt(np.sum((clean - x) ** 2) / np.sum(clean ** 2)))


def correlation(clean, x) -> float:
    """Coeficiente de correlação de Pearson."""
    clean, x = _align(clean, x)
    return float(np.corrcoef(clean, x)[0, 1])


def summary(clean, noisy, filtered) -> dict:
    """Dicionário com todas as métricas para uma comparação (limpo/ruidoso/filtrado)."""
    return {
        "snr_in_db": snr_db(clean, noisy),
        "snr_out_db": snr_db(clean, filtered),
        "snr_improvement_db": snr_improvement(clean, noisy, filtered),
        "rmse": rmse(clean, filtered),
        "prd": prd(clean, filtered),
        "corr": correlation(clean, filtered),
    }
