"""Pipeline de dados para treino/validação/teste da CNN-DAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import tensorflow as tf

from src.data_io import FS, ensure_datasets, load_ecg, load_noise
from src.noise.contaminate import SNR_TARGETS, add_noise_at_snr, powerline_60hz

# 38 registros → treino + validação interna
TRAIN_RECORDS: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219",
)

# 10 registros → avaliação final (nunca vistos durante treino/validação)
TEST_RECORDS: tuple[str, ...] = (
    "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
)

_ALL_MITDB = frozenset(TRAIN_RECORDS) | frozenset(TEST_RECORDS)
assert len(_ALL_MITDB) == 48, "Contagem incorreta de registros MIT-BDB."
assert frozenset(TRAIN_RECORDS).isdisjoint(TEST_RECORDS), "Vazamento treino/teste!"

NOISE_TYPES_REAL = ("bw", "ma", "em")
POWERLINE_FREQS  = {"60hz": 60.0, "50hz": 50.0}
ALL_NOISE_TYPES  = NOISE_TYPES_REAL + tuple(POWERLINE_FREQS)

WINDOW        = 1024
TRAIN_OVERLAP = 0.75
TRAIN_STRIDE  = int(WINDOW * (1 - TRAIN_OVERLAP))

VAL_FRACTION         = 0.20
NOISE_TEST_FRACTION  = 0.20

SNR_TRAIN_RANGE = (-2.0, 20.0)
EPS = 1e-8

def split_signal(
    signal: np.ndarray,
    val_fraction: float = VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide um sinal de TREINO em bloco de treino (início) e validação (final)."""

    n = len(signal)
    cut = int(n * (1 - val_fraction))
    return signal[:cut], signal[cut:]


def split_noise(
    noise: np.ndarray,
    test_fraction: float = NOISE_TEST_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide um arquivo de ruído em pool de treino/val (início) e teste (final)."""

    n = len(noise)
    cut = int(n * (1 - test_fraction))
    return noise[:cut], noise[cut:]

def windowize(
    signal: np.ndarray,
    window: int = WINDOW,
    stride: int = TRAIN_STRIDE,
) -> np.ndarray:
    n = len(signal)
    if n < window:
        return np.empty((0, window), dtype=np.float64)
    starts = np.arange(0, n - window + 1, stride)
    return np.stack([signal[s : s + window] for s in starts], axis=0)


def _noise_segment(
    noise_pool: np.ndarray,
    window: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sorteia um trecho de ``window`` amostras dentro de ``noise_pool``."""
    max_offset = len(noise_pool) - window
    if max_offset <= 0:
        reps = int(np.ceil(window / len(noise_pool)))
        return np.tile(noise_pool, reps)[:window]
    offset = int(rng.integers(0, max_offset + 1))
    return noise_pool[offset : offset + window]


def contaminate_window(
    clean_window: np.ndarray,
    noise_type: str,
    snr_db: float,
    noise_pools: dict,
    rng: np.random.Generator,
    fs: int = FS,
) -> np.ndarray:
    """Contamina uma janela limpa com ``noise_type`` no ``snr_db`` alvo."""

    window = len(clean_window)
    if noise_type in POWERLINE_FREQS:
        noise = powerline_60hz(
            window,
            fs=fs,
            f0=POWERLINE_FREQS[noise_type],
            phase=float(rng.uniform(0, 2 * np.pi)),
        )
    elif noise_type in NOISE_TYPES_REAL:
        noise = _noise_segment(noise_pools[noise_type], window, rng)
    else:
        raise ValueError(f"noise_type desconhecido: {noise_type!r}")
    return add_noise_at_snr(clean_window, noise, snr_db, rng=rng)

def zscore_pair(
    clean_window: np.ndarray,
    noisy_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza (limpo, ruidoso) com média/desvio do LIMPO."""

    mu = clean_window.mean()
    sigma = clean_window.std() + EPS
    return (noisy_window - mu) / sigma, (clean_window - mu) / sigma


@dataclass
class DataPools:
    """Blocos de dados já cortados e janelados, prontos para os geradores."""

    ecg_train_windows: np.ndarray
    ecg_val_windows:   np.ndarray
    ecg_test_windows:  np.ndarray
    noise_train:       dict
    noise_test:        dict


def build_data_pools(
    train_records: tuple[str, ...] = TRAIN_RECORDS,
    test_records:  tuple[str, ...] = TEST_RECORDS,
    window:              int   = WINDOW,
    train_stride:        int   = TRAIN_STRIDE,
    val_fraction:        float = VAL_FRACTION,
    noise_test_fraction: float = NOISE_TEST_FRACTION,
) -> DataPools:
    """Carrega tudo do disco e aplica os splits. Chamar uma única vez antes do treino."""
    
    ensure_datasets()

    # ECG de treino e validação (38 registros)
    train_chunks, val_chunks = [], []
    for rec in train_records:
        clean, fs = load_ecg(rec)
        assert fs == FS, f"registro {rec} com fs inesperado: {fs}"
        tr, va = split_signal(clean, val_fraction)
        train_chunks.append(windowize(tr, window, train_stride))
        val_chunks.append(windowize(va, window, train_stride))
    ecg_train_windows = np.concatenate(train_chunks, axis=0)
    ecg_val_windows   = np.concatenate(val_chunks,   axis=0)

    # ECG de teste (10 registros — inteiros, sem split temporal)
    test_chunks = []
    for rec in test_records:
        clean, fs = load_ecg(rec)
        assert fs == FS, f"registro de teste {rec} com fs inesperado: {fs}"
        test_chunks.append(windowize(clean, window, window))  # sem overlap
    ecg_test_windows = np.concatenate(test_chunks, axis=0)

    # Ruído real: split 80% treino-val / 20% teste
    noise_train, noise_test = {}, {}
    for nt in NOISE_TYPES_REAL:
        raw = load_noise(nt)
        tr, te = split_noise(raw, noise_test_fraction)
        noise_train[nt] = tr
        noise_test[nt]  = te

    return DataPools(
        ecg_train_windows=ecg_train_windows,
        ecg_val_windows=ecg_val_windows,
        ecg_test_windows=ecg_test_windows,
        noise_train=noise_train,
        noise_test=noise_test,
    )


def _make_generator(
    ecg_windows: np.ndarray,
    noise_pools: dict,
    seed: int,
    snr_range: tuple[float, float] = SNR_TRAIN_RANGE,
    noise_types: tuple = ALL_NOISE_TYPES,
):
    rng      = np.random.default_rng(seed)
    n_windows = len(ecg_windows)
    n_types   = len(noise_types)

    def gen() -> Iterator[tuple]:
        while True:
            idx        = rng.integers(0, n_windows)
            clean_w    = ecg_windows[idx]
            noise_type = noise_types[rng.integers(0, n_types)]
            snr        = float(rng.uniform(*snr_range))
            noisy_w    = contaminate_window(clean_w, noise_type, snr, noise_pools, rng)
            x, y       = zscore_pair(clean_w, noisy_w)
            yield x.astype(np.float32)[:, None], y.astype(np.float32)[:, None]

    return gen


def make_train_dataset(
    pools: DataPools,
    batch_size: int = 32,
    seed: int = 42,
) -> tf.data.Dataset:
    """Dataset de treino: janelas dos 38 registros de treino + pool de ruído de TREINO."""

    gen = _make_generator(pools.ecg_train_windows, pools.noise_train, seed=seed)
    sig = (
        tf.TensorSpec(shape=(WINDOW, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(WINDOW, 1), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig) \
                          .batch(batch_size) \
                          .prefetch(tf.data.AUTOTUNE)


def make_val_dataset(
    pools: DataPools,
    batch_size: int = 32,
    seed: int = 123,
) -> tf.data.Dataset:
    """Dataset de validação: janelas de val dos 38 registros + pool de ruído de TREINO."""
    
    gen = _make_generator(pools.ecg_val_windows, pools.noise_train, seed=seed)
    sig = (
        tf.TensorSpec(shape=(WINDOW, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(WINDOW, 1), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig) \
                          .batch(batch_size) \
                          .prefetch(tf.data.AUTOTUNE)


def build_test_table(
    pools: DataPools,
    noise_types: tuple = ALL_NOISE_TYPES,
    snr_targets: tuple = SNR_TARGETS,
    seed: int = 2026,
) -> dict:
    """Gera pares (ruidoso, limpo) determinísticos para avaliação final."""
    
    rng = np.random.default_rng(seed)
    table = {}
    for noise_type in noise_types:
        for snr in snr_targets:
            noisy_list, clean_list = [], []
            for clean_w in pools.ecg_test_windows:
                noisy_w = contaminate_window(
                    clean_w, noise_type, float(snr), pools.noise_test, rng
                )
                x, y = zscore_pair(clean_w, noisy_w)
                noisy_list.append(x)
                clean_list.append(y)
            table[(noise_type, snr)] = (
                np.stack(noisy_list)[..., None].astype(np.float32),
                np.stack(clean_list)[..., None].astype(np.float32),
            )
    return table


if __name__ == "__main__":
    pools = build_data_pools()

    print("=== ECG ===")
    print(f"  janelas treino : {pools.ecg_train_windows.shape}  "
          f"({len(TRAIN_RECORDS)} registros, 80% de cada)")
    print(f"  janelas val    : {pools.ecg_val_windows.shape}  "
          f"({len(TRAIN_RECORDS)} registros, 20% de cada)")
    print(f"  janelas teste  : {pools.ecg_test_windows.shape}  "
          f"({len(TEST_RECORDS)} registros inteiros)")

    print("\n=== Ruído ===")
    for nt in NOISE_TYPES_REAL:
        print(f"  {nt}: treino={len(pools.noise_train[nt])}  "
              f"teste={len(pools.noise_test[nt])}")

    print("\n=== Batch de treino ===")
    train_ds = make_train_dataset(pools, batch_size=8)
    xb, yb = next(iter(train_ds))
    print(f"  x={xb.shape}  y={yb.shape}")

    print("\n=== Tabela de teste ===")
    table = build_test_table(pools)
    for (nt, snr), (X, Y) in list(table.items())[:3]:
        print(f"  ({nt}, {snr:+.0f} dB): X={X.shape}  Y={Y.shape}")
