#!/usr/bin/env bash
# Install BlackHole, the virtual audio device SilentDictate plays into on macOS.
set -euo pipefail

if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

brew install --cask blackhole-2ch

cat <<'EOF'

BlackHole 2ch is installed. REBOOT before using it; macOS loads audio drivers at startup.

After rebooting:
  1. In the app that should hear SilentDictate, choose "BlackHole 2ch" as its microphone.
  2. Apps with no input picker, such as Claude Code /voice, use only the default device.
     For those, set System Settings > Sound > Input to "BlackHole 2ch". Restore your real
     mic there afterwards, and set SilentDictate's own input with input_device.
  3. Set output_mode = "virtual_mic" (or "both") in the SilentDictate config.
  4. Check it with: silentdictate doctor
EOF
