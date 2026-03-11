#!/bin/bash
# setup_sumo.sh — Install SUMO on Ubuntu/Debian
set -e

echo "======================================"
echo " NeuroTraffic-RL — SUMO Setup Script"
echo "======================================"

# Add SUMO PPA
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update -q
sudo apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO_HOME
SUMO_PATH=$(sumo --version 2>&1 | head -1 | grep -oP '/[^ ]+' | head -1 || echo "")
if [ -z "$SUMO_PATH" ]; then
    SUMO_HOME="/usr/share/sumo"
else
    SUMO_HOME=$(dirname "$(dirname "$SUMO_PATH")")
fi

echo "export SUMO_HOME=\"$SUMO_HOME\"" >> ~/.bashrc
export SUMO_HOME="$SUMO_HOME"

echo ""
echo "✅ SUMO installed successfully."
echo "   SUMO_HOME = $SUMO_HOME"
echo "   sumo version: $(sumo --version 2>&1 | head -1)"
echo ""
echo "Run: source ~/.bashrc  (or restart your terminal)"
echo "Then: make train"
