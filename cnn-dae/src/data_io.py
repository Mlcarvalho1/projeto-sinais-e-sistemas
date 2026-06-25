"""Acesso aos dados do PhysioNet para a CNN-DAE.

Estende o data_io original do projeto ao cobrir todos os 48 registros do
MIT-BIH Arrhythmia Database (mitdb) — necessários para o split de 38
registros de treino/val + 10 de teste.

O data_io original do projeto usa apenas os 5 registros de DEFAULT_RECORDS
para os filtros convencionais. Aqui, ALL_RECORDS cobre o banco completo.
O acesso ao nstdb (ruído bw/ma/em) é idêntico ao original.

Sinal lido via ``wfdb`` a 360 Hz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

# ---------------------------------------------------------------------------
# Frequência de amostragem comum a mitdb e nstdb (Hz)
# ---------------------------------------------------------------------------
FS: int = 360

# ---------------------------------------------------------------------------
# Todos os 48 registros do MIT-BIH Arrhythmia Database
# Nota: a numeração tem saltos — não existem 110, 120 etc.
# ---------------------------------------------------------------------------
ALL_RECORDS: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)
assert len(ALL_RECORDS) == 48, "Lista de registros MIT-BDB incompleta."

# ---------------------------------------------------------------------------
# Registros de ruído real do nstdb
# bw = baseline wander | ma = muscle artifact | em = electrode motion
# ---------------------------------------------------------------------------
NOISE_RECORDS: tuple[str, ...] = ("bw", "ma", "em")

# ---------------------------------------------------------------------------
# Caminhos — relativos à raiz do subprojeto cnn_dae/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]   # cnn_dae/
DATA_DIR  = _ROOT / "data"
MITDB_DIR = DATA_DIR / "mitdb"
NSTDB_DIR = DATA_DIR / "nstdb"


def download_mitdb(
    records: tuple | list = ALL_RECORDS,
    dest: Path = MITDB_DIR,
) -> Path:
    """Baixa registros de ECG do mitdb para ``dest`` (idempotente)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(dest), records=list(records))
    return dest


def download_nstdb(
    noises: tuple | list = NOISE_RECORDS,
    dest: Path = NSTDB_DIR,
) -> Path:
    """Baixa registros de ruído do nstdb para ``dest`` (idempotente)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("nstdb", str(dest), records=list(noises))
    return dest


def ensure_datasets(
    records: tuple | list = ALL_RECORDS,
    noises: tuple | list = NOISE_RECORDS,
) -> None:
    """Garante que todos os registros necessários estão baixados localmente.

    Idempotente: só faz download dos arquivos que ainda não existem em disco.
    Baixa os 48 registros do mitdb + os 3 arquivos de ruído do nstdb.
    """
    missing_ecg = [r for r in records if not (MITDB_DIR / f"{r}.dat").exists()]
    if missing_ecg:
        print(f"Baixando {len(missing_ecg)} registro(s) do mitdb: {missing_ecg}")
        download_mitdb(missing_ecg)

    missing_noise = [n for n in noises if not (NSTDB_DIR / f"{n}.dat").exists()]
    if missing_noise:
        print(f"Baixando {len(missing_noise)} arquivo(s) de ruído do nstdb: {missing_noise}")
        download_nstdb(missing_noise)


def load_ecg(
    record_id: str | int,
    channel: int = 0,
    dest: Path = MITDB_DIR,
) -> tuple[np.ndarray, int]:
    """Carrega um canal de ECG do mitdb.

    Parâmetros
    ----------
    record_id : str ou int
        Número do registro (ex: ``"100"`` ou ``100``).
    channel : int
        Canal a carregar (0 = MLII na maioria dos registros).

    Retorna
    -------
    sinal : np.ndarray, shape (N,)
    fs    : int — frequência de amostragem em Hz
    """
    if str(record_id) not in ALL_RECORDS:
        raise ValueError(
            f"Registro {record_id!r} não pertence ao MIT-BDB. "
            f"Use um dos 48 registros em ALL_RECORDS."
        )
    rec = wfdb.rdrecord(str(Path(dest) / str(record_id)))
    sig = rec.p_signal[:, channel].astype(np.float64)
    return sig, int(rec.fs)


def load_noise(
    noise_type: str,
    channel: int = 0,
    dest: Path = NSTDB_DIR,
) -> np.ndarray:
    """Carrega um canal de ruído real do nstdb (``bw``, ``ma`` ou ``em``)."""
    if noise_type not in NOISE_RECORDS:
        raise ValueError(
            f"ruído desconhecido: {noise_type!r} (use {NOISE_RECORDS})"
        )
    rec = wfdb.rdrecord(str(Path(dest) / noise_type))
    return rec.p_signal[:, channel].astype(np.float64)


if __name__ == "__main__":
    print(f"Verificando/baixando {len(ALL_RECORDS)} registros MIT-BDB + nstdb...")
    ensure_datasets()
    print("Download concluído.\n")

    for rec_id in ALL_RECORDS[:3]:
        sig, fs = load_ecg(rec_id)
        print(
            f"  registro {rec_id}: {len(sig)} amostras  fs={fs} Hz  "
            f"min={sig.min():.3f}  max={sig.max():.3f}"
        )

    print()
    for nt in NOISE_RECORDS:
        noise = load_noise(nt)
        print(f"  ruído {nt}: {len(noise)} amostras")
