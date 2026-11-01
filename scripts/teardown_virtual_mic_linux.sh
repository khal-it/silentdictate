#!/usr/bin/env bash
# Remove the SilentDictate virtual mic and any persistent config.
set -euo pipefail

pactl list short modules | awk '/silentdict/ {print $1}' | tac | while read -r id; do
  pactl unload-module "$id"
done

conf="${XDG_CONFIG_HOME:-$HOME/.config}/pipewire/pipewire.conf.d/90-silentdictate.conf"
if [[ -f "$conf" ]]; then
  rm "$conf"
  echo "Removed $conf. Restart PipeWire: systemctl --user restart pipewire pipewire-pulse"
fi
echo "Virtual mic removed."
