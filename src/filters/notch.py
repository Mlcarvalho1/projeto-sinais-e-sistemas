"""Filtro Notch IIR de 60 Hz — rejeição da interferência da rede elétrica.

Estrutura IIR de 2ª ordem (biquad) projetada por :func:`scipy.signal.iirnotch`.
O fator de qualidade ``Q = f0 / bw`` controla a largura de banda -3 dB.
"""
from __future__ import annotations

from scipy.signal import filtfilt, iirnotch

FS = 360


def design_notch(f0: float = 60.0, fs: int = FS, bw: float = 2.0):
    """Projeta o notch. ``bw`` é a banda -3 dB em Hz (Q = f0/bw). Retorna ``(b, a)``."""
    Q = f0 / bw
    b, a = iirnotch(w0=f0, Q=Q, fs=fs)
    return b, a


def apply_notch(signal, f0: float = 60.0, fs: int = FS, bw: float = 2.0):
    """Aplica o notch com fase zero (``filtfilt``)."""
    b, a = design_notch(f0, fs, bw)
    return filtfilt(b, a, signal)
