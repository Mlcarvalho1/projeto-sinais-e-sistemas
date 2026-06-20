"""Filtro Passa-Baixa FIR (janela de Hamming) — atenuação de ruído muscular (EMG).

FIR de fase linear projetado por :func:`scipy.signal.firwin`. A janela de Hamming
dá atenuação de lóbulo lateral de ~-41 dB. Por ser FIR de fase linear, o atraso de
grupo é constante e igual a ``(numtaps - 1) / 2`` amostras.
"""
from __future__ import annotations

from scipy.signal import filtfilt, firwin, lfilter

FS = 360


def design_lowpass_fir(fc: float = 40.0, fs: int = FS, numtaps: int = 61, window: str = "hamming"):
    """Projeta o FIR passa-baixa. Retorna o vetor de coeficientes (taps)."""
    return firwin(numtaps, cutoff=fc, window=window, fs=fs)


def group_delay_samples(numtaps: int = 61) -> float:
    """Atraso de grupo (constante) de um FIR de fase linear: ``(numtaps - 1) / 2``."""
    return (numtaps - 1) / 2.0


def apply_lowpass_fir(signal, fc: float = 40.0, fs: int = FS, numtaps: int = 61,
                      window: str = "hamming", zero_phase: bool = False):
    """Aplica o FIR. ``zero_phase=False`` usa ``lfilter`` (causal, atraso (M-1)/2);
    ``True`` usa ``filtfilt`` (sem atraso, mas não causal)."""
    taps = design_lowpass_fir(fc, fs, numtaps, window)
    return filtfilt(taps, [1.0], signal) if zero_phase else lfilter(taps, 1.0, signal)
