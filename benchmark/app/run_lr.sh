#!/bin/bash

# Linear Regression Benchmark Script
# Single-thread mode for reproducible results
export OMP_NUM_THREADS=1

RESULT_FILE="lr_benchmark_results.txt"

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

# Initialize result file
cat > "$RESULT_FILE" << EOL
==============================================================================
  Linear Regression Benchmark Results
  Date: $(date)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
==============================================================================

EOL

run_app() {
    local name=$1
    local exec=$2

    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    if [ -f "$exec" ]; then
        echo "Cooling down before $name..."
        sleep 30

        OMP_NUM_THREADS=1 ./$exec 2>&1 | tee -a "$RESULT_FILE"

        echo ""
        echo "Cooling down after $name..."
        sleep 30
    else
        echo "Warning: $exec not found, skipping..."
    fi
}

# Run plaintext baseline
run_app "Plaintext Baseline" "lr_plaintext"

# Run encrypted comparison (Naive vs NewCol vs AR24)
run_app "Encrypted (Naive/NewCol/AR24)" "lr_benchmark"

echo ""
echo "=============================================="
echo "  Linear Regression Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $BUILD_DIR/$RESULT_FILE"
