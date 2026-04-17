#!/bin/bash
# juice_cuda_test.sh
# Authenticates with Juice, connects to a pool, and runs the CUDA test
# inside the Apptainer container.
#
# Usage:
#   ./juice_cuda_test.sh                        # interactive pool selection
#   ./juice_cuda_test.sh <pool-id>              # use a specific pool ID
#   JUICE_POOL_ID=<pool-id> ./juice_cuda_test.sh

set -euo pipefail

JUICE="${JUICE_BIN:-./juice/juice}"
SIF="${SIF_PATH:-./cuda_test.sif}"
POOL_ID="${1:-${JUICE_POOL_ID:-}}"
JUICE_DIR="$(dirname $(realpath $JUICE))"

# --- Sanity checks ----------------------------------------------------------

if [ ! -x "$JUICE" ]; then
    echo "ERROR: juice binary not found at '$JUICE'"
    echo "Set JUICE_BIN=/path/to/juice or run from the cuda_test directory."
    exit 1
fi

if [ ! -f "$SIF" ]; then
    echo "ERROR: SIF file not found at '$SIF'"
    echo "Build it first with:"
    echo "  apptainer build cuda_test.sif cuda_test.def"
    echo "Or set SIF_PATH=/path/to/cuda_test.sif"
    exit 1
fi

if ! command -v apptainer &>/dev/null; then
    echo "ERROR: apptainer not found on PATH"
    exit 1
fi

# --- Login ------------------------------------------------------------------

echo "==> Checking Juice authentication..."
if ! "$JUICE" pool list &>/dev/null; then
    echo "==> Not logged in. Starting Juice login..."
    "$JUICE" login
else
    echo "==> Already authenticated."
fi

# --- Pool selection ---------------------------------------------------------

if [ -z "$POOL_ID" ]; then
    echo ""
    echo "==> Available pools:"
    "$JUICE" pool list
    echo ""
    read -rp "Enter pool ID to use: " POOL_ID
fi

echo "==> Using pool: $POOL_ID"

# --- Apptainer Run ----------------------------------------------------------
# Juice uses LD_PRELOAD to inject libcuda.so, which has dependencies
# on its own libraries in the Juice directory. We bind the entire Juice
# directory into the container at /juice and set LD_LIBRARY_PATH so the
# dynamic linker can find everything.

echo ""
echo "==> Launching CUDA test inside Apptainer via Juice..."
echo ""

APPTAINERENV_LD_LIBRARY_PATH=/juice:/juice/compute:/juice/graphics \
"$JUICE" run --pool-ids "$POOL_ID" \
    apptainer run \
    --bind "$JUICE_DIR":/juice \
    "$SIF"

