#!/bin/bash
# Usage: ./build.sh [target_env]
# Example: ./build.sh uno_r4_wifi
# Example: ./build.sh portenta_h7
# If no target specified, builds both environments

TARGET_ENV="$1"

#buf generate
cd embedded

if [ -z "$TARGET_ENV" ]; then
  # No target specified, build both environments
  pio run -e portenta_h7 -e uno_r4_wifi -t compiledb
  pio run -e portenta_h7 -e uno_r4_wifi || { echo 'Build failed'; exit 1; }
else
  # Specific target environment
  pio run -e "$TARGET_ENV" -t compiledb
  pio run -e "$TARGET_ENV" || { echo 'Build failed'; exit 1; }
fi
