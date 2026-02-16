#!/bin/bash

# Multi-thread mode for faster benchmarks
# export OMP_NUM_THREADS=1  # Uncomment for single-thread reproducible benchmarks

echo "Running matrix inversion benchmarks (multi-thread)..."
echo "Starting at $(date)"

# Initialize result file with header
cat > inversion_benchmark_results.txt << EOL
Matrix Inversion Performance Benchmarks
$(date)
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
    ./$algo 2>&1 | tee -a inversion_benchmark_results.txt

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
    # "benchmark_inversion_newrow"
    "benchmark_inversion_ar24"
    "benchmark_inversion_jkls18"
    "benchmark_inversion_rt22"
    # "benchmark_inversion_diag"
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