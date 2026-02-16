#!/bin/bash

# Linear Regression Benchmark Script
# Single-thread mode for reproducible results
export OMP_NUM_THREADS=1

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

run_app() {
    local name=$1
    local exec=$2

    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    if [ -f "$exec" ]; then
        echo "Cooling down before $name..."
        sleep 30

        OMP_NUM_THREADS=1 ./$exec

        echo ""
        echo "Cooling down after $name..."
        sleep 30
    else
        echo "Warning: $exec not found, skipping..."
    fi
}

# Run plaintext baseline
run_app "Plaintext Baseline" "lr_plaintext"

# Run encrypted versions
run_app "Encrypted (NewCol)" "lr_newcol"
run_app "Encrypted (AR24)" "lr_ar24"

echo ""
echo "=============================================="
echo "  Linear Regression Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"

# Generate app summary if all benchmarks are done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_app_summary.sh" ]; then
    echo ""
    echo "Generating app summary..."
    bash "$SCRIPT_DIR/generate_app_summary.sh"
fi
