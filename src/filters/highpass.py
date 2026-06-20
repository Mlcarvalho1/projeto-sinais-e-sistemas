"""Filtro Passa-Alta IIR Butterworth — correção da variação de linha de base.

Butterworth de 2ª ordem (magnitude maximamente plana na banda de passagem),
projetado por :func:`scipy.signal.butter` via transformação bilinear (interna).
Aplicado com ``filtfilt`` (fase zero) ou ``lfilter`` (causal) para comparação.
"""
from __future__ import annotations

from scipy.signal import butter, filtfilt, lfilter

FS = 360


def design_highpass(fc: float = 0.5, fs: int = FS, order: int = 2):
    """Projeta o passa-alta Butterworth. Retorna ``(b, a)``."""
    b, a = butter(order, fc, btype="highpass", fs=fs)
    return b, a


def apply_highpass(signal, fc: float = 0.5, fs: int = FS, order: int = 2, zero_phase: bool = True):
    """Aplica o passa-alta. ``zero_phase=True`` usa ``filtfilt``; senão ``lfilter`` (causal)."""
    b, a = design_highpass(fc, fs, order)
    return filtfilt(b, a, signal) if zero_phase else lfilter(b, a, signal)
