from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class SignalAnalysisTool:
    """Extract lightweight time- and frequency-domain evidence from a 1-D signal."""

    def analyze(self, signal: Iterable[float], sampling_rate: float = 1024.0) -> dict[str, Any]:
        x = np.asarray(list(signal), dtype=np.float64).reshape(-1)
        if x.size < 4:
            raise ValueError("signal must contain at least four samples")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        centered = x - x.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate)

        # Ignore DC when finding the dominant oscillatory component.
        if spectrum.size > 1:
            dominant_idx = int(np.argmax(spectrum[1:]) + 1)
        else:
            dominant_idx = 0

        power = spectrum**2
        total_power = float(power.sum()) + 1e-12
        nyquist = sampling_rate / 2.0
        high_freq_mask = freqs >= 0.5 * nyquist
        high_freq_ratio = float(power[high_freq_mask].sum() / total_power)

        rms = float(np.sqrt(np.mean(x**2)))
        peak = float(np.max(np.abs(x)))
        crest_factor = float(peak / (rms + 1e-12))

        return {
            "num_samples": int(x.size),
            "sampling_rate_hz": float(sampling_rate),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "rms": rms,
            "peak": peak,
            "peak_to_peak": float(np.ptp(x)),
            "crest_factor": crest_factor,
            "dominant_frequency_hz": float(freqs[dominant_idx]),
            "dominant_magnitude": float(spectrum[dominant_idx]),
            "high_frequency_energy_ratio": high_freq_ratio,
        }
