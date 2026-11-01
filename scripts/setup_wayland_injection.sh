#!/usr/bin/env bash
# Install Wayland text injection tools and uinput access for ydotool.
#
# WARNING: membership of the input group lets every program you run read and inject input
# events, including keystrokes. On a shared machine, prefer wtype or wl-copy instead.
set -euo pipefail

if command -v apt-get >/dev/null; then
  sudo apt-get install -y wtype ydotool wl-clipboard
elif command -v dnf >/dev/null; then
  sudo dnf install -y wtype ydotool wl-clipboard
elif command -v pacman >/dev/null; then
  sudo pacman -S --needed wtype ydotool wl-clipboard
else
  echo "Unknown package manager. Install wtype, ydotool and wl-clipboard yourself." >&2
fi

read -r -p "Grant your user /dev/uinput access for ydotool? See the warning above. [y/N] " reply
if [[ "$reply" =~ ^[Yy]$ ]]; then
  echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' |
    sudo tee /etc/udev/rules.d/80-silentdictate-uinput.rules >/dev/null
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  sudo usermod -aG input "$USER"

  unit="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/ydotoold.service"
  if [[ ! -f /usr/lib/systemd/user/ydotool.service && ! -f "$unit" ]]; then
    mkdir -p "$(dirname "$unit")"
    cat >"$unit" <<'EOF'
[Unit]
Description=ydotool daemon

[Service]
ExecStart=/usr/bin/ydotoold
Restart=on-failure

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now ydotoold.service
  elif [[ -f /usr/lib/systemd/user/ydotool.service ]]; then
    systemctl --user enable --now ydotool.service
  fi
  echo "Log out and back in so the input group takes effect."
fi

cat <<'EOF'

Bind these in your compositor, since Wayland has no global hotkeys for apps:
  silentdictate toggle       (tap mode)
  silentdictate ptt-start    (key press)
  silentdictate ptt-stop     (key release)

Sway example:
  bindsym --no-repeat Control_R exec silentdictate ptt-start
  bindsym --release Control_R exec silentdictate ptt-stop

Note: wtype works on Sway and Hyprland but not GNOME. GNOME uses ydotool or wl-copy.
Check with: silentdictate doctor
EOF
