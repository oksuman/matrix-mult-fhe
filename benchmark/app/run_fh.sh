#!/bin/bash

# Fixed Hessian Logistic Regression Benchmark Script
# Single-thread mode for reproducible results
export OMP_NUM_THREADS=1

RESULT_FILE="fh_benchmark_results.txt"

echo "=============================================="
echo "  Fixed Hessian Benchmark"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/fixed-hessian"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

# Initialize result file
cat > "$RESULT_FILE" << EOL
==============================================================================
  Fixed Hessian Benchmark Results
  Date: $(date)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
  Dataset: Diabetes (64 train, 256 test)
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

# Run plaintext baseline (Diabetes only)
run_app "Plaintext (Diabetes)" "fh_plaintext_diabetes"

# Run encrypted version (Diabetes only)
run_app "Encrypted (Diabetes)" "fh_encrypted_diabetes"

echo ""
echo "=============================================="
echo "  Fixed Hessian Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved to: $BUILD_DIR/$RESULT_FILE"
