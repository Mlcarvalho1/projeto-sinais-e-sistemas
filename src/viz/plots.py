"""Visualizações para formalização e validação dos filtros.

Diagrama de polos/zeros, |H(e^jω)| (resposta em magnitude), atraso de grupo,
espectro FFT antes/depois e sobreposição no tempo. Cada função aceita um ``ax``
opcional para compor figuras maiores.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, group_delay, tf2zpk

FS = 360


def plot_pole_zero(b, a, ax=None, title="Diagrama de polos e zeros"):
    z, p, _ = tf2zpk(b, a)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8)
    ax.scatter(z.real, z.imag, s=80, facecolors="none", edgecolors="C0", label="zeros")
    ax.scatter(p.real, p.imag, s=80, marker="x", color="C3", label="polos")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_freq_response(b, a=1.0, fs=FS, ax=None, title="Resposta em frequência |H(e^jω)|"):
    w, h = freqz(b, a, worN=4096, fs=fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_group_delay(b, a=1.0, fs=FS, ax=None, title="Atraso de grupo"):
    w, gd = group_delay((np.atleast_1d(b), np.atleast_1d(a)), w=4096, fs=fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(w, gd)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Atraso de grupo (amostras)")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_spectrum(x, fs=FS, ax=None, label=None, title="Espectro (FFT)"):
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x * np.hanning(len(x)))
    freq = np.fft.rfftfreq(len(x), 1 / fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq, 20 * np.log10(np.abs(spec) + 1e-12), label=label)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True)
    if label:
        ax.legend()
    return ax


def plot_time_overlay(clean=None, noisy=None, filtered=None, fs=FS, n=None, ax=None,
                      title="Sinais no tempo"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    series = [
        ("ruidoso", noisy, {"color": "0.7", "lw": 0.8}),
        ("limpo", clean, {"color": "C0", "lw": 1.0}),
        ("filtrado", filtered, {"color": "C3", "lw": 1.0}),
    ]
    for label, sig, style in series:
        if sig is None:
            continue
        s = np.asarray(sig, dtype=np.float64)
        if n:
            s = s[:n]
        t = np.arange(len(s)) / fs
        ax.plot(t, s, label=label, **style)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax
