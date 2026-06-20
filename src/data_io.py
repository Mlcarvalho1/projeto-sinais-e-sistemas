"""Acesso aos dados do PhysioNet: MIT-BIH Arrhythmia (mitdb) e Noise Stress Test (nstdb).

Sinal de ECG de referência ("limpo") vem do mitdb; o ruído fisiológico real
(bw, ma, em) vem do nstdb. Tudo é lido via ``wfdb`` a 360 Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

#: Frequência de amostragem comum a mitdb e nstdb (Hz).
FS = 360

#: Registros de ECG sugeridos na Entrega 1 / TODO.
DEFAULT_RECORDS = ("100", "103", "105", "115", "215")

#: Registros de ruído real do nstdb (baseline wander, muscle artifact, electrode motion).
NOISE_RECORDS = ("bw", "ma", "em")

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data"
MITDB_DIR = DATA_DIR / "mitdb"
NSTDB_DIR = DATA_DIR / "nstdb"


def download_mitdb(records=DEFAULT_RECORDS, dest=MITDB_DIR) -> Path:
    """Baixa os registros de ECG do mitdb para ``dest`` (idempotente)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(dest), records=list(records))
    return dest


def download_nstdb(noises=NOISE_RECORDS, dest=NSTDB_DIR) -> Path:
    """Baixa os registros de ruído do nstdb para ``dest`` (idempotente)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("nstdb", str(dest), records=list(noises))
    return dest


def ensure_datasets(records=DEFAULT_RECORDS, noises=NOISE_RECORDS) -> None:
    """Garante que mitdb (records) e nstdb (noises) estão baixados localmente."""
    missing_ecg = [r for r in records if not (MITDB_DIR / f"{r}.dat").exists()]
    if missing_ecg:
        download_mitdb(missing_ecg)
    missing_noise = [n for n in noises if not (NSTDB_DIR / f"{n}.dat").exists()]
    if missing_noise:
        download_nstdb(missing_noise)


def load_ecg(record_id, channel: int = 0, dest=MITDB_DIR):
    """Carrega um canal de ECG do mitdb. Retorna ``(sinal: np.ndarray, fs: int)``."""
    rec = wfdb.rdrecord(str(Path(dest) / str(record_id)))
    sig = rec.p_signal[:, channel].astype(np.float64)
    return sig, int(rec.fs)


def load_noise(noise_type: str, channel: int = 0, dest=NSTDB_DIR) -> np.ndarray:
    """Carrega um canal de ruído real do nstdb (``bw``, ``ma`` ou ``em``)."""
    if noise_type not in NOISE_RECORDS:
        raise ValueError(f"ruído desconhecido: {noise_type!r} (use {NOISE_RECORDS})")
    rec = wfdb.rdrecord(str(Path(dest) / noise_type))
    return rec.p_signal[:, channel].astype(np.float64)
