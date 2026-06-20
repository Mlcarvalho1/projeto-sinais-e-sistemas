"""Smoke tests offline (não baixam dados do PhysioNet)."""
import numpy as np
import pytest

from src.noise.contaminate import add_noise_at_snr, contaminate, powerline_60hz
from src.filters.notch import design_notch, apply_notch
from src.filters.highpass import design_highpass
from src.filters.lowpass_fir import design_lowpass_fir, group_delay_samples
from src.metrics import metrics


def _synthetic_ecg(n=4000, fs=360):
    """Sinal sintético simples com energia em baixa frequência (proxy de ECG)."""
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 5 * t)


@pytest.mark.parametrize("snr", [0, 6, 12, 18])
def test_add_noise_hits_target_snr(snr):
    clean = _synthetic_ecg()
    noise = np.random.default_rng(0).standard_normal(len(clean))
    noisy = add_noise_at_snr(clean, noise, snr)
    assert metrics.snr_db(clean, noisy) == pytest.approx(snr, abs=1e-6)


def test_contaminate_60hz_hits_target_snr():
    clean = _synthetic_ecg()
    noisy = contaminate(clean, "60hz", snr_db=6)
    assert metrics.snr_db(clean, noisy) == pytest.approx(6, abs=1e-6)


def test_powerline_has_60hz_peak():
    n, fs = 3600, 360
    x = powerline_60hz(n, fs=fs)
    freq = np.fft.rfftfreq(n, 1 / fs)
    mag = np.abs(np.fft.rfft(x))
    assert freq[np.argmax(mag)] == pytest.approx(60.0, abs=0.5)


def test_filter_shapes():
    b, a = design_notch()
    assert len(b) == 3 and len(a) == 3
    b, a = design_highpass(order=2)
    assert len(b) == 3 and len(a) == 3
    taps = design_lowpass_fir(numtaps=61)
    assert len(taps) == 61
    assert group_delay_samples(61) == 30.0


def test_notch_attenuates_60hz():
    clean = _synthetic_ecg()
    noisy = contaminate(clean, "60hz", snr_db=0)
    filtered = apply_notch(noisy)
    # energia do filtrado fica muito mais próxima do limpo
    assert metrics.snr_improvement(clean, noisy, filtered) > 10


def test_metrics_identity():
    x = _synthetic_ecg()
    assert metrics.prd(x, x) == pytest.approx(0.0, abs=1e-9)
    assert metrics.correlation(x, x) == pytest.approx(1.0, abs=1e-9)
    assert metrics.rmse(x, x) == pytest.approx(0.0, abs=1e-9)
