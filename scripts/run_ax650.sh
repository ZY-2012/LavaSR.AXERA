#!/bin/bash
# LavaSR v2 AX650 one-click demo.
#   bash run_ax650.sh [input.wav] [output.wav] [--cpp|--python] [--no-denoise]
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1

INPUT="${1:-samples/libri_16k.wav}"
OUTPUT="${2:-output.wav}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi

MODE="cpp"
EXTRA=()
for a in "$@"; do
  case "$a" in
    --python) MODE="python" ;;
    --cpp) MODE="cpp" ;;
    *) EXTRA+=("$a") ;;
  esac
done
if [ ! -x ./bin/lavasr_ax650 ]; then MODE="python"; fi

if [ "$MODE" = "cpp" ]; then
  exec ./bin/lavasr_ax650 models/den_encoder.axmodel models/den_decoder.axmodel \
      models/enhancer.axmodel "$INPUT" "$OUTPUT" "${EXTRA[@]}"
else
  exec python3 python/demo.py "$INPUT" -o "$OUTPUT" --models models "${EXTRA[@]}"
fi
