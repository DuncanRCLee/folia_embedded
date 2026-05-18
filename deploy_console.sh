#!/bin/bash
# Usage: ./deploy_console.sh <target_env>
# Example: ./deploy_console.sh uno_r4_wifi

TARGET_ENV="$1"

if [ -z "$TARGET_ENV" ]; then
  echo "Error: No target environment specified."
  echo "Usage: $0 <target_env>"
  exit 1
fi

./build.sh
cd embedded
echo "running"
pio run -e "$TARGET_ENV" --target upload || { echo 'Upload failed'; exit 1; }

# Wait a moment for the board to reboot and port to reappear
sleep 1

# Find the most recently modified /dev/cu.usbmodem* port
PORT=$(ls -1t /dev/cu.usbmodem* 2>/dev/null | head -n 1)

if [ -z "$PORT" ]; then
  echo "No /dev/cu.usbmodem* device found!"
  exit 1
fi

echo "Connecting to serial port: $PORT"
pio device monitor -b 115200 -p "$PORT"
