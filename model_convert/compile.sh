#!/bin/bash
# Compile the three LavaSR submodels with Pulsar2 8.0.
#   PULSAR2_IMAGE=pulsar2:8.0
#   HASP_HOME=<driverless-license-dir>   # only needed for the driverless license
set -e
cd "$(dirname "$0")/.."
IMAGE="${PULSAR2_IMAGE:-pulsar2:8.0}"
HASP_ARGS=(-v /var/hasplm:/var/hasplm)
if [ -n "${HASP_HOME:-}" ]; then
  HASP_ARGS+=(-v "${HASP_HOME}:/root/.hasplm")
fi
for name in den_encoder den_decoder enhancer; do
  echo "=== compile ${name} ==="
  docker run --rm --network host \
    -v "$PWD:/workspace" \
    "${HASP_ARGS[@]}" \
    -e HASP_HOME=/root/.hasplm \
    "$IMAGE" -lc "set -e; mkdir -p /root/.hasplm/installed/32434 && cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null || true; PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:\$PATH; pulsar2 build --config /workspace/model_convert/configs/${name}.json"
done
ls -lh build/compile/*/model.axmodel
