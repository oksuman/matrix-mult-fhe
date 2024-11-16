#!/bin/bash

echo "Running matrix inversion benchmarks..."
echo "Starting at $(date)"

# Initialize result file
cat > inversion_benchmark_results.txt << EOL
Matrix Inversion Performance Benchmarks
$(date)
-----------------------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------
EOL

run_benchmark() {
    local algo=$1
    echo "Running $algo..."
    
    # Cooling period before benchmark
    echo "System cooling before $algo..."
    sleep 15
    sync
    
    # Run benchmark and capture output
    ./$algo | tee >(grep "BM_" | grep -v "^Running" >> inversion_benchmark_results.txt)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        return 1
    fi
    
    # Cooling period after benchmark
    echo "System cooling after $algo..."
    sleep 15
    
    echo "" >> inversion_benchmark_results.txt
}

# List of inversion benchmark executables
INVERSION_ALGORITHMS=(
    "benchmark_inversion_jkls18"
    "benchmark_inversion_rt22"
    "benchmark_inversion_as24"
    "benchmark_inversion_newcol"
    "benchmark_inversion_newrow"
    "benchmark_inversion_diag"
)

# Run each benchmark
for algo in "${INVERSION_ALGORITHMS[@]}"; do
    run_benchmark $algo
done

# Add footer to results file
echo "-----------------------------------------------------------------------------------------------" >> inversion_benchmark_results.txt
echo "Benchmarks completed at $(date)" >> inversion_benchmark_results.txt

echo "Inversion benchmarks complete. Results saved in inversion_benchmark_results.txt"