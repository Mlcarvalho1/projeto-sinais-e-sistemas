"""Gera os notebooks 01/02/03 via nbformat (garante JSON válido).

    ./.venv/bin/python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = (
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path.cwd().parent))  # raiz do repo no path\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "%matplotlib inline"
)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, str(NB / name))
    print("escrito:", name)


# ----------------------------------------------------------------------------- 01
write("01_dados.ipynb", [
    md("# 01 — Dados: ECG (mitdb), ruídos (nstdb) e contaminação controlada\n\n"
       "Objetivo: baixar os dados do PhysioNet, visualizar o ECG de referência e os "
       "ruídos reais, e demonstrar a função `contaminate` em SNRs conhecidos."),
    code(SETUP),
    code("from src.data_io import ensure_datasets, load_ecg, load_noise, FS\n"
         "from src.noise.contaminate import contaminate, powerline_60hz, SNR_TARGETS\n"
         "from src.metrics.metrics import snr_db\n"
         "ensure_datasets()  # baixa mitdb (100,103,105,115,215) + nstdb (bw,ma,em)"),
    md("## ECG de referência (registro 100)"),
    code("clean, fs = load_ecg('100')\n"
         "print(f'{len(clean)} amostras @ {fs} Hz  (~{len(clean)/fs/60:.1f} min)')\n"
         "t = np.arange(5*fs)/fs\n"
         "plt.figure(figsize=(11,3)); plt.plot(t, clean[:5*fs]); plt.xlabel('s'); "
         "plt.ylabel('mV'); plt.title('ECG limpo — registro 100 (5 s)'); plt.grid(True)"),
    md("## Ruídos reais do nstdb + senoide de 60 Hz"),
    code("fig, axs = plt.subplots(2, 2, figsize=(12,5))\n"
         "for ax, nz in zip(axs.ravel(), ['bw','ma','em']):\n"
         "    s = load_noise(nz); ax.plot(s[:5*FS]); ax.set_title(nz); ax.grid(True)\n"
         "axs.ravel()[3].plot(powerline_60hz(5*FS)); axs.ravel()[3].set_title('60 Hz'); "
         "axs.ravel()[3].grid(True); fig.tight_layout()"),
    md("## Contaminação em SNRs alvo (0/6/12/18 dB)\n\n"
       "Verificamos que o SNR medido bate com o alvo (validação da escala)."),
    code("for snr in SNR_TARGETS:\n"
         "    noisy = contaminate(clean, '60hz', snr_db=snr)\n"
         "    print(f'alvo {snr:2d} dB -> medido {snr_db(clean, noisy):.2f} dB')"),
    code("fig, axs = plt.subplots(len(SNR_TARGETS), 1, figsize=(11,8), sharex=True)\n"
         "for ax, snr in zip(axs, SNR_TARGETS):\n"
         "    noisy = contaminate(clean, 'ma', snr_db=snr)\n"
         "    ax.plot(noisy[:3*FS]); ax.set_ylabel(f'{snr} dB'); ax.grid(True)\n"
         "axs[0].set_title('ECG contaminado com ruído muscular (ma) por SNR'); fig.tight_layout()"),
])

# ----------------------------------------------------------------------------- 02
write("02_filtros_convencionais.ipynb", [
    md("# 02 — Filtros convencionais: projeto, validação e formalização\n\n"
       "Notch IIR 60 Hz · Passa-alta IIR Butterworth 0,5 Hz · Passa-baixa FIR Hamming 40 Hz.\n"
       "Gera as figuras de formalização (polos/zeros, |H(e^jω)|, atraso de grupo, FFT) "
       "usadas no relatório."),
    code(SETUP),
    code("from src.data_io import ensure_datasets, load_ecg, FS\n"
         "from src.noise.contaminate import contaminate\n"
         "from src.filters.notch import design_notch, apply_notch\n"
         "from src.filters.highpass import design_highpass, apply_highpass\n"
         "from src.filters.lowpass_fir import design_lowpass_fir, apply_lowpass_fir, group_delay_samples\n"
         "from src.metrics.metrics import summary\n"
         "from src.viz import plots\n"
         "ensure_datasets(); clean, fs = load_ecg('100')"),
    md("## 1. Notch IIR 60 Hz (`iirnotch`, Q = f0/BW = 60/2 = 30)"),
    code("b, a = design_notch(f0=60, fs=fs, bw=2)\n"
         "fig, axs = plt.subplots(1, 2, figsize=(12,4))\n"
         "plots.plot_pole_zero(b, a, ax=axs[0]); plots.plot_freq_response(b, a, fs=fs, ax=axs[1])\n"
         "fig.tight_layout()"),
    code("noisy = contaminate(clean, '60hz', 6); filt = apply_notch(noisy)\n"
         "fig, ax = plt.subplots(figsize=(8,4))\n"
         "plots.plot_spectrum(noisy[:3600], fs=fs, ax=ax, label='ruidoso')\n"
         "plots.plot_spectrum(filt[:3600], fs=fs, ax=ax, label='filtrado')\n"
         "print(summary(clean, noisy, filt))"),
    md("## 2. Passa-alta IIR Butterworth 0,5 Hz (`butter` + `filtfilt`)\n\n"
       "**Nota de protocolo:** o ECG do mitdb já contém conteúdo <0,5 Hz; para avaliar a "
       "remoção da deriva de forma justa, comparamos contra `ref = apply_highpass(clean)`."),
    code("b, a = design_highpass(fc=0.5, fs=fs, order=2)\n"
         "fig, axs = plt.subplots(1, 2, figsize=(12,4))\n"
         "plots.plot_pole_zero(b, a, ax=axs[0]); plots.plot_freq_response(b, a, fs=fs, ax=axs[1])\n"
         "axs[1].set_xlim(0, 5); fig.tight_layout()"),
    code("ref = apply_highpass(clean)\n"
         "noisy = contaminate(clean, 'bw', 6); filt = apply_highpass(noisy)\n"
         "noisy_ref = noisy - clean + ref\n"
         "print(summary(ref, noisy_ref, filt))\n"
         "plots.plot_time_overlay(clean=ref, noisy=noisy_ref, filtered=filt, fs=fs, n=3600)"),
    md("## 3. Passa-baixa FIR Hamming 40 Hz (`firwin`, 61 taps)\n\n"
       "FIR de fase linear → atraso de grupo constante `(M-1)/2 = 30` amostras. "
       "Avaliar em fase-zero (`filtfilt`) ou compensar o atraso."),
    code("taps = design_lowpass_fir(fc=40, fs=fs, numtaps=61, window='hamming')\n"
         "print('atraso de grupo:', group_delay_samples(61), 'amostras')\n"
         "fig, axs = plt.subplots(1, 3, figsize=(15,4))\n"
         "plots.plot_pole_zero(taps, [1.0], ax=axs[0])\n"
         "plots.plot_freq_response(taps, fs=fs, ax=axs[1])\n"
         "plots.plot_group_delay(taps, [1.0], fs=fs, ax=axs[2]); fig.tight_layout()"),
    code("noisy = contaminate(clean, 'ma', 6)\n"
         "filt = apply_lowpass_fir(noisy, zero_phase=True)\n"
         "print(summary(clean, noisy, filt))\n"
         "# Δ SNR ≈ 0 contra ruído muscular banda-larga: limitação que motiva os avançados."),
    md("## 4. Tabela comparativa (protocolo correto)"),
    code("import importlib.util, pathlib\n"
         "spec = importlib.util.spec_from_file_location('gf', pathlib.Path.cwd().parent/'scripts'/'gerar_figuras.py')\n"
         "gf = importlib.util.module_from_spec(spec); spec.loader.exec_module(gf)\n"
         "gf.tabela_comparativa('100')"),
])

# ----------------------------------------------------------------------------- 03
write("03_demo.ipynb", [
    md("# 03 — Demo interativa: filtrando ECG ao vivo\n\n"
       "Escolha o registro, o tipo de ruído, o SNR e o filtro; o painel mostra o sinal "
       "no tempo, o espectro (FFT) e as métricas em tempo real. (Requer `ipywidgets`.)"),
    code(SETUP),
    code("import ipywidgets as widgets\n"
         "from IPython.display import display\n"
         "from src.data_io import ensure_datasets, load_ecg, DEFAULT_RECORDS, FS\n"
         "from src.noise.contaminate import contaminate\n"
         "from src.filters.notch import apply_notch\n"
         "from src.filters.highpass import apply_highpass\n"
         "from src.filters.lowpass_fir import apply_lowpass_fir\n"
         "from src.metrics.metrics import summary\n"
         "from src.viz import plots\n"
         "ensure_datasets()"),
    code("FILTROS = {\n"
         "    'Notch 60Hz': lambda y: apply_notch(y),\n"
         "    'Passa-alta 0.5Hz': lambda y: apply_highpass(y),\n"
         "    'Passa-baixa FIR 40Hz': lambda y: apply_lowpass_fir(y, zero_phase=True),\n"
         "}\n"
         "_cache = {}\n"
         "def _load(rec):\n"
         "    if rec not in _cache: _cache[rec] = load_ecg(rec)\n"
         "    return _cache[rec]\n"
         "\n"
         "def atualizar(record, ruido, snr, filtro):\n"
         "    clean, fs = _load(record)\n"
         "    ref = apply_highpass(clean) if ruido == 'bw' else clean\n"
         "    noisy = contaminate(clean, ruido, snr_db=snr)\n"
         "    noisy_ref = noisy - clean + ref\n"
         "    filt = FILTROS[filtro](noisy)\n"
         "    fig, axs = plt.subplots(1, 2, figsize=(13,4))\n"
         "    plots.plot_time_overlay(clean=ref, noisy=noisy_ref, filtered=filt, fs=fs, n=3*fs, ax=axs[0])\n"
         "    plots.plot_spectrum(noisy_ref[:6*fs], fs=fs, ax=axs[1], label='ruidoso')\n"
         "    plots.plot_spectrum(filt[:6*fs], fs=fs, ax=axs[1], label='filtrado')\n"
         "    m = summary(ref, noisy_ref, filt)\n"
         "    print(f\"SNR {m['snr_in_db']:+.1f} -> {m['snr_out_db']:+.1f} dB (Δ{m['snr_improvement_db']:+.1f}) | \"\n"
         "          f\"PRD {m['prd']:.1f}% | corr {m['corr']:.3f}\")\n"
         "    plt.show()"),
    code("widgets.interact(\n"
         "    atualizar,\n"
         "    record=widgets.Dropdown(options=list(DEFAULT_RECORDS), value='100', description='registro'),\n"
         "    ruido=widgets.Dropdown(options=['60hz','bw','ma','em'], value='60hz', description='ruído'),\n"
         "    snr=widgets.IntSlider(min=0, max=24, step=2, value=6, description='SNR dB'),\n"
         "    filtro=widgets.Dropdown(options=list(FILTROS), value='Notch 60Hz', description='filtro'),\n"
         ");"),
])
print("OK")
