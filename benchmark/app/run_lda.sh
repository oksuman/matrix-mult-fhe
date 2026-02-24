#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_lda.sh
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

LOG_FILE="lda_console.log"

echo "=============================================="
echo "  LDA Benchmark"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/linear-discriminant-analysis"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

# Initialize log file
cat > "$LOG_FILE" << EOL
==============================================================================
  LDA Benchmark Console Log
  Date: $(date)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
==============================================================================

EOL

run_app() {
    local name=$1
    local exec=$2
    shift 2
    local args="$@"

    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    if [ -f "$exec" ]; then
        echo "Cooling down before $name..."
        sleep 30

        ./$exec $args 2>&1 | tee -a "$LOG_FILE"

        echo ""
        echo "Cooling down after $name..."
        sleep 30
    else
        echo "Warning: $exec not found, skipping..."
    fi
}

# Run plaintext baseline (--no-save: no separate txt file)
run_app "Plaintext Baseline" "lda_plaintext" --no-save

# Run encrypted version
run_app "Encrypted LDA" "lda_encrypted" --benchmark

# Run benchmark version (writes lda_results.txt)
run_app "Benchmark LDA" "benchmark_lda"

echo ""
echo "=============================================="
echo "  LDA Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Console log: $BUILD_DIR/$LOG_FILE"
echo "Result file: $BUILD_DIR/lda_results.txt"
