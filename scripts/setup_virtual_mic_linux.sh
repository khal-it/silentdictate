#!/usr/bin/env bash
# Create the SilentDictate virtual mic on PipeWire or PulseAudio.
# Modules last until logout; pass --persist to also install a PipeWire config.
set -euo pipefail

if ! command -v pactl >/dev/null; then
  echo "pactl is required (pulseaudio-utils or pipewire-pulse)." >&2
  exit 1
fi

if pactl list short sinks | grep -q 'silentdict-sink'; then
  echo "silentdict-sink already exists."
else
  pactl load-module module-null-sink media.class=Audio/Sink sink_name=silentdict-sink channel_map=mono
  pactl load-module module-remap-source master=silentdict-sink.monitor source_name=silentdict-mic source_properties=device.description=SilentDictate_Mic
fi

if [[ "${1:-}" == "--persist" ]]; then
  conf="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire.conf.d/90-silentdictate.conf"
  mkdir -p "$(dirname "$conf")"
  cat >"$conf" <<'EOF'
context.objects = [
  { factory = adapter
    args = {
      factory.name     = support.null-audio-sink
      node.name        = "silentdict-sink"
      media.class      = Audio/Sink
      audio.position   = [ MONO ]
    }
  }
  { factory = adapter
    args = {
      factory.name     = support.null-audio-sink
      node.name        = "silentdict-mic"
      node.description = "SilentDictate_Mic"
      media.class      = Audio/Source/Virtual
      audio.position   = [ MONO ]
    }
  }
]
context.modules = [
  { name = libpipewire-module-loopback
    args = {
      capture.props  = { node.target = "silentdict-sink" stream.capture.sink = true }
      playback.props = { node.target = "silentdict-mic" }
    }
  }
]
EOF
  echo "Wrote $conf. Restart PipeWire: systemctl --user restart pipewire pipewire-pulse"
fi

cat <<'EOF'

Select "SilentDictate_Mic" as the microphone in the receiving app, or make it the default:
  pactl set-default-source silentdict-mic
SilentDictate plays into silentdict-sink, which is the default virtual_mic_device on Linux.
Remove it with scripts/teardown_virtual_mic_linux.sh.
EOF
