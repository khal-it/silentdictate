"""SilentDictate: whisper-quiet local dictation."""

import os

__version__ = "0.1.0"

# No telemetry, per the spec's hard constraints.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
