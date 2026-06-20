"""Contaminação controlada de ECG.

Soma a um sinal de ECG "limpo" um ruído real do nstdb (``bw``, ``ma``, ``em``) ou
uma senoide de 60 Hz (interferência da rede), escalado para atingir um SNR alvo.
É o "coração do estudo comparativo": gera pares (limpo, contaminado) com SNR
conhecido, permitindo métricas objetivas (ver :mod:`src.metrics.metrics`).
"""
from __future__ import annotations

import numpy as np

from ..data_io import FS, load_noise

#: Níveis de SNR alvo do protocolo (dB).
SNR_TARGETS = (0, 6, 12, 18)

_REAL_NOISES = ("bw", "ma", "em")
_PLI_ALIASES = ("60hz", "pli", "powerline")


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def _match_length(noise: np.ndarray, n: int, rng=None) -> np.ndarray:
    """Recorta (com offset opcional) ou replica o ruído até o comprimento ``n``."""
    noise = np.asarray(noise, dtype=np.float64)
    if len(noise) == n:
        return noise
    if len(noise) > n:
        start = 0 if rng is None else int(rng.integers(0, len(noise) - n + 1))
        return noise[start:start + n]
    reps = int(np.ceil(n / len(noise)))
    return np.tile(noise, reps)[:n]


def add_noise_at_snr(clean, noise, snr_db: float, rng=None) -> np.ndarray:
    """Retorna ``clean + k*noise`` com ``k`` ajustado para o SNR alvo (dB).

    ``k = sqrt(P_clean / (P_noise * 10**(snr/10)))``.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = _match_length(noise, len(clean), rng=rng)
    p_noise = _power(noise)
    if p_noise == 0.0:
        raise ValueError("ruído com potência nula")
    k = np.sqrt(_power(clean) / (p_noise * 10 ** (snr_db / 10.0)))
    return clean + k * noise


def powerline_60hz(n: int, fs: int = FS, f0: float = 60.0, phase: float = 0.0) -> np.ndarray:
    """Senoide de 60 Hz (interferência da rede elétrica)."""
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f0 * t + phase)


def contaminate(signal, noise_type: str, snr_db: float, fs: int = FS, rng=None) -> np.ndarray:
    """Contamina ``signal`` com ``noise_type`` (``bw``/``ma``/``em``/``60hz``) no ``snr_db`` alvo."""
    signal = np.asarray(signal, dtype=np.float64)
    nt = noise_type.lower()
    if nt in _REAL_NOISES:
        noise = load_noise(nt)
    elif nt in _PLI_ALIASES:
        noise = powerline_60hz(len(signal), fs=fs)
    else:
        raise ValueError(f"noise_type desconhecido: {noise_type!r}")
    return add_noise_at_snr(signal, noise, snr_db, rng=rng)
