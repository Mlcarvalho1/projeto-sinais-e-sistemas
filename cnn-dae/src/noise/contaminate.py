"""Contaminação controlada de ECG para a CNN-DAE.

Soma a um sinal de ECG "limpo" um ruído real do nstdb (``bw``, ``ma``, ``em``)
ou uma senoide de rede elétrica (60 Hz / 50 Hz), escalado para atingir um
SNR alvo em dB. Gera os pares (ruidoso, limpo) com SNR conhecido que
alimentam o treino e a avaliação da CNN-DAE.

Nota: este módulo é independente do ``contaminate.py`` dos filtros
convencionais — lida com 5 tipos de ruído e SNR contínuo no treino.
"""
from __future__ import annotations

import numpy as np

from src.data_io import FS, load_noise

# ---------------------------------------------------------------------------
# Protocolo de avaliação — 4 níveis discretos de SNR (dB)
# ---------------------------------------------------------------------------
SNR_TARGETS: tuple[int, ...] = (0, 6, 12, 18)

_REAL_NOISES = ("bw", "ma", "em")
_PLI_ALIASES = ("60hz", "pli", "powerline")


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _power(x: np.ndarray) -> float:
    """Potência média do sinal."""
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def _match_length(noise: np.ndarray, n: int, rng=None) -> np.ndarray:
    """Recorta (com offset aleatório opcional) ou replica o ruído até o comprimento ``n``."""
    noise = np.asarray(noise, dtype=np.float64)
    if len(noise) == n:
        return noise
    if len(noise) > n:
        start = 0 if rng is None else int(rng.integers(0, len(noise) - n + 1))
        return noise[start : start + n]
    reps = int(np.ceil(n / len(noise)))
    return np.tile(noise, reps)[:n]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def add_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng=None,
) -> np.ndarray:
    """Retorna ``clean + k*noise`` com ``k`` ajustado para o SNR alvo (dB).

    ``k = sqrt(P_clean / (P_noise * 10^(snr/10)))``.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noise = _match_length(noise, len(clean), rng=rng)
    p_noise = _power(noise)
    if p_noise == 0.0:
        raise ValueError("ruído com potência nula — verifique o arquivo do nstdb.")
    k = np.sqrt(_power(clean) / (p_noise * 10 ** (snr_db / 10.0)))
    return clean + k * noise


def powerline_60hz(
    n: int,
    fs: int = FS,
    f0: float = 60.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Gera uma senoide de interferência de rede elétrica (padrão 60 Hz ou 50 Hz).

    Parâmetros
    ----------
    n     : número de amostras
    fs    : frequência de amostragem em Hz (padrão: 360)
    f0    : frequência da interferência (60.0 ou 50.0 Hz)
    phase : fase inicial em radianos (use fase aleatória no treino)
    """
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f0 * t + phase)


def contaminate(
    signal: np.ndarray,
    noise_type: str,
    snr_db: float,
    fs: int = FS,
    rng=None,
) -> np.ndarray:
    """Contamina ``signal`` com ``noise_type`` no ``snr_db`` alvo.

    Parâmetros
    ----------
    signal     : sinal ECG limpo, shape (N,)
    noise_type : ``"bw"`` | ``"ma"`` | ``"em"`` | ``"60hz"`` | ``"50hz"``
    snr_db     : SNR alvo em dB
    fs         : frequência de amostragem (padrão: 360 Hz)
    rng        : gerador numpy para offset aleatório no ruído real

    Retorna
    -------
    sinal contaminado, shape (N,)
    """
    signal = np.asarray(signal, dtype=np.float64)
    nt = noise_type.lower()
    if nt in _REAL_NOISES:
        noise = load_noise(nt)
    elif nt in _PLI_ALIASES or nt == "50hz":
        f0 = 50.0 if nt == "50hz" else 60.0
        noise = powerline_60hz(len(signal), fs=fs, f0=f0)
    else:
        raise ValueError(
            f"noise_type desconhecido: {noise_type!r}  "
            f"(use bw | ma | em | 60hz | 50hz)"
        )
    return add_noise_at_snr(signal, noise, snr_db, rng=rng)
