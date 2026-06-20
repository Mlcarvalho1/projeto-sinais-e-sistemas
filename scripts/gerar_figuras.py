"""Gera as figuras de formalização e a tabela comparativa dos filtros convencionais.

Saídas em ``figures/`` (PNG) e ``figures/comparacao_convencionais.csv``. Usa o
protocolo de avaliação correto (FIR em fase-zero; referência passa-alta para a
deriva de linha de base). Execute com o venv principal:

    ./.venv/bin/python scripts/gerar_figuras.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz do repo no path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_io import FS, ensure_datasets, load_ecg
from src.filters.highpass import apply_highpass, design_highpass
from src.filters.lowpass_fir import apply_lowpass_fir, design_lowpass_fir, group_delay_samples
from src.filters.notch import apply_notch, design_notch
from src.metrics.metrics import summary
from src.noise.contaminate import SNR_TARGETS, contaminate
from src.viz import plots

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)


def formalizacao_filtro(nome, b, a, *, is_fir=False):
    """Painel polos/zeros + |H| + atraso de grupo de um filtro."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plots.plot_pole_zero(b, a, ax=axs[0], title=f"{nome} — polos/zeros")
    plots.plot_freq_response(b, a, fs=FS, ax=axs[1], title=f"{nome} — |H(e^jω)|")
    plots.plot_group_delay(b, a, fs=FS, ax=axs[2], title=f"{nome} — atraso de grupo")
    fig.tight_layout()
    out = FIG / f"formalizacao_{nome.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def antes_depois(nome, clean, noisy, filtered, fs=FS, n=1800):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    plots.plot_time_overlay(clean=clean, noisy=noisy, filtered=filtered, fs=fs, n=n, ax=axs[0],
                            title=f"{nome} — tempo (5 s)")
    plots.plot_spectrum(noisy[:n*2], fs=fs, ax=axs[1], label="ruidoso", title=f"{nome} — FFT")
    plots.plot_spectrum(filtered[:n*2], fs=fs, ax=axs[1], label="filtrado")
    fig.tight_layout()
    out = FIG / f"antes_depois_{nome.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def tabela_comparativa(record="100"):
    """Tabela filtros × ruído × SNR com o protocolo de avaliação correto."""
    clean, fs = load_ecg(record)
    ref_hp = apply_highpass(clean)  # referência consistente p/ deriva de linha de base
    seg = slice(1000, len(clean) - 1000)
    rows = []
    plano = [
        ("Notch 60Hz", "60hz", lambda y: apply_notch(y), clean),
        ("Passa-alta 0.5Hz", "bw", lambda y: apply_highpass(y), ref_hp),
        ("Passa-baixa FIR 40Hz", "ma", lambda y: apply_lowpass_fir(y, zero_phase=True), clean),
        ("Passa-baixa FIR 40Hz", "em", lambda y: apply_lowpass_fir(y, zero_phase=True), clean),
    ]
    for nome, ruido, filt, ref in plano:
        for snr in SNR_TARGETS:
            noisy = contaminate(clean, ruido, snr_db=snr)
            noisy_ref = noisy - clean + ref  # mesma referência que 'ref'
            filtered = filt(noisy)
            m = summary(ref[seg], noisy_ref[seg], filtered[seg])
            rows.append({"filtro": nome, "ruido": ruido, "snr_in_alvo": snr,
                         "snr_out_db": round(m["snr_out_db"], 2),
                         "delta_snr_db": round(m["snr_improvement_db"], 2),
                         "prd_pct": round(m["prd"], 2), "corr": round(m["corr"], 3)})
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "comparacao_convencionais.csv", index=False)
    return df


def main():
    ensure_datasets()
    clean, fs = load_ecg("100")

    # 1) Formalização dos três filtros
    bn, an = design_notch()
    formalizacao_filtro("Notch 60Hz", bn, an)
    bh, ah = design_highpass()
    formalizacao_filtro("Passa-alta 0.5Hz", bh, ah)
    taps = design_lowpass_fir()
    formalizacao_filtro("Passa-baixa FIR 40Hz", taps, [1.0], is_fir=True)
    print(f"atraso de grupo do FIR: {group_delay_samples(len(taps))} amostras")

    # 2) Antes/depois (tempo + FFT)
    antes_depois("Notch 60Hz", clean, contaminate(clean, "60hz", 6), apply_notch(contaminate(clean, "60hz", 6)))
    noisy_bw = contaminate(clean, "bw", 6)
    antes_depois("Passa-alta 0.5Hz", apply_highpass(clean), noisy_bw, apply_highpass(noisy_bw))
    noisy_ma = contaminate(clean, "ma", 6)
    antes_depois("Passa-baixa FIR 40Hz", clean, noisy_ma, apply_lowpass_fir(noisy_ma, zero_phase=True))

    # 3) Tabela comparativa
    df = tabela_comparativa("100")
    print(df.to_string(index=False))
    print(f"\nFiguras e CSV em: {FIG}")


if __name__ == "__main__":
    main()
