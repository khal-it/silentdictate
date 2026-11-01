"""Audio preprocessing: high-pass and peak normalization; never gates whispers out."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .audio_in import SAMPLE_RATE, Audio

HIGHPASS_HZ = 90.0
TARGET_PEAK_DBFS = -3.0
MAX_GAIN_DB = 70.0
_SOS = signal.butter(4, HIGHPASS_HZ, "highpass", fs=SAMPLE_RATE, output="sos")


def preprocess(audio: Audio) -> Audio:
    """Remove rumble, then normalize the peak to about -3 dBFS."""
    if audio.size == 0:
        return audio
    filtered: Audio = signal.sosfilt(_SOS, audio).astype(np.float32)
    peak = float(np.max(np.abs(filtered)))
    if peak == 0:
        return filtered
    gain = min(10 ** (TARGET_PEAK_DBFS / 20) / peak, 10 ** (MAX_GAIN_DB / 20))
    return np.asarray(filtered * gain, dtype=np.float32)
