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
Parameters: MULT_DEPTH=36, SCALE_MOD_SIZE=59, FIRST_MOD_SIZE=60
Bootstrapping: levelBudget={4,4}, numIterations=2, precision=18
Scalar inverse iterations: 1
Inversion iterations: d=4:18, d=8:22, d=16:25, d=32:27, d=64:30
Seed: 1000 + run
-----------------------------------------------------------------------------------------------
EOL

run_benchmark() {
    local algo=$1
    echo ""
    echo "========================================"
    echo "  Running $algo"
    echo "========================================"

    # Cooling period before benchmark
    echo "System cooling (45s)..."
    sleep 45
    sync

    # Run benchmark and save output
    ./$algo $NUM_RUNS 2>&1 | tee -a inversion_benchmark_results.txt

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        return 1
    fi

    # Cooling period after benchmark
    echo ""
    echo "Cooling down (45s)..."
    sleep 45
}

# List of inversion benchmark executables
INVERSION_ALGORITHMS=(
    "benchmark_inversion_newcol"
    "benchmark_inversion_ar24"
    "benchmark_inversion_jkls18"
    "benchmark_inversion_rt22"
    "benchmark_inversion_naive"
)

# Run each benchmark
for algo in "${INVERSION_ALGORITHMS[@]}"; do
    run_benchmark $algo
done

# Add footer to results file
echo "Benchmarks completed at $(date)" >> inversion_benchmark_results.txt

echo "Inversion benchmarks complete. Results saved in inversion_benchmark_results.txt"

# Generate summary tables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_summary_table.sh" ]; then
    echo ""
    echo "Generating summary tables..."
    zsh "$SCRIPT_DIR/generate_summary_table.sh"
fi