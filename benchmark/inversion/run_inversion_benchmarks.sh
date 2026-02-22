#!/bin/bash

# Single-thread mode for reproducible benchmarks
export OMP_NUM_THREADS=1

NUM_RUNS=${1:-1}

echo "Running matrix inversion benchmarks (single-thread)..."
echo "Starting at $(date)"
echo "Runs per dimension: $NUM_RUNS"

# Initialize result file with header
cat > inversion_benchmark_results.txt << EOL
Matrix Inversion Performance Benchmarks
$(date)
Trials: $NUM_RUNS
OMP_NUM_THREADS: $OMP_NUM_THREADS
Parameters: MULT_DEPTH=30, SCALE_MOD_SIZE=59, FIRST_MOD_SIZE=60
Bootstrapping: levelBudget={4,4}, numIterations=2, precision=18
Scalar inverse iterations: 1
Inversion iterations: d=4:18, d=8:22, d=16:25, d=32:27, d=64:30
Seed: 1000 + run
-----------------------------------------------------------------------------------------------
EOL

COOLING_PERIOD=45

run_benchmark() {
    local algo=$1
    local dim=$2
    echo ""
    echo "========================================"
    echo "  Running $algo d=$dim"
    echo "========================================"

    # Cooling period before benchmark
    echo "System cooling (${COOLING_PERIOD}s)..."
    sleep $COOLING_PERIOD
    sync

    # Run benchmark for a single dimension in its own process
    ./$algo $NUM_RUNS $dim 2>&1 | tee -a inversion_benchmark_results.txt

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo d=$dim"
        echo "[ERROR] $algo d=$dim failed" >> inversion_benchmark_results.txt
        return 1
    fi
}

# --- NewCol: d=4,8,16,32,64 ---
for dim in 4 8 16 32 64; do
    run_benchmark "benchmark_inversion_newcol" $dim
done

# --- AR24: d=4,8,16,32,64 ---
for dim in 4 8 16 32 64; do
    run_benchmark "benchmark_inversion_ar24" $dim
done

# --- JKLS18: d=4,8,16,32,64 ---
for dim in 4 8 16 32 64; do
    run_benchmark "benchmark_inversion_jkls18" $dim
done

# --- RT22: d=4,8,16,32 (no d=64) ---
for dim in 4 8 16 32; do
    run_benchmark "benchmark_inversion_rt22" $dim
done

# --- Naive: d=4,8 only ---
for dim in 4 8; do
    run_benchmark "benchmark_inversion_naive" $dim
done

# Add footer to results file
echo "Benchmarks completed at $(date)" >> inversion_benchmark_results.txt

echo ""
echo "Inversion benchmarks complete. Results saved in inversion_benchmark_results.txt"

# Generate summary tables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_summary_table.sh" ]; then
    echo ""
    echo "Generating summary tables..."
    zsh "$SCRIPT_DIR/generate_summary_table.sh"
fi
