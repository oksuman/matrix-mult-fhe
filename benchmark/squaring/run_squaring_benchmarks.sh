#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_squaring_benchmarks.sh
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

NUM_RUNS=${1:-1}

echo "============================================"
echo "  Matrix Squaring Benchmarks"
echo "============================================"
echo "Runs per dimension: $NUM_RUNS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

run_benchmark() {
    local algo=$1
    echo ""
    echo "========================================"
    echo "  Running $algo"
    echo "========================================"

    echo "System cooling (45s)..."
    sleep 45
    sync

    ./$algo $NUM_RUNS 2>&1 | tee -a squaring_console.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        return 1
    fi

    echo ""
    echo "Cooling down (45s)..."
    sleep 45
}

# List of squaring benchmark executables
SQUARING_ALGORITHMS=(
    "benchmark_squaring_naive"
    "benchmark_squaring_newcol"
    "benchmark_squaring_ar24"
    # "benchmark_squaring_newrow"
    "benchmark_squaring_jkls18"
    "benchmark_squaring_rt22"
    # "benchmark_squaring_diag"
)

# Initialize result file
cat > squaring_console.log << EOL
============================================
  Matrix Squaring Benchmark Results
============================================
Date: $(date)
Trials: $NUM_RUNS
OMP_NUM_THREADS: $OMP_NUM_THREADS
Parameters: SCALE_MOD_SIZE=50, FIRST_MOD_SIZE=60
Squaring iterations: 15

EOL

# Run each benchmark
for algo in "${SQUARING_ALGORITHMS[@]}"; do
    run_benchmark $algo
done

echo "" >> squaring_console.log
echo "============================================" >> squaring_console.log
echo "  All Benchmarks Complete" >> squaring_console.log
echo "============================================" >> squaring_console.log

echo ""
echo "============================================"
echo "  All Benchmarks Complete"
echo "============================================"
echo "Results saved in squaring_console.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_summary_table.sh" ]; then
    echo ""
    echo "Generating summary tables..."
    zsh "$SCRIPT_DIR/generate_summary_table.sh"
fi
