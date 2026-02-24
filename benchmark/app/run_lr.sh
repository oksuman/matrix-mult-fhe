#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_lr.sh
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

LOG_FILE="lr_console.log"

echo "=============================================="
echo "  Linear Regression Benchmark"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/linear-regression"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

# Initialize log file
cat > "$LOG_FILE" << EOL
==============================================================================
  Linear Regression Benchmark Console Log
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

# Run plaintext baseline
run_app "Plaintext Baseline" "lr_plaintext"

# Run encrypted comparison (Naive vs NewCol vs AR24, writes lr_results.txt)
run_app "Encrypted (Naive/NewCol/AR24)" "lr_benchmark" --benchmark

echo ""
echo "=============================================="
echo "  Linear Regression Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Console log: $BUILD_DIR/$LOG_FILE"
echo "Result file: $BUILD_DIR/lr_results.txt"
