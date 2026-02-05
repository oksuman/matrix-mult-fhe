#!/bin/bash

echo "Running matrix squaring benchmarks..."
sleep 2

run_benchmark() {
    local algo=$1
    echo "Running $algo..."
    
    # Cooling period before benchmark
    echo "System cooling before $algo..."
    sleep 45
    sync
    
    # Run benchmark and capture output
    ./$algo | tee >(grep "BM_" | grep -v "^Running" >> squaring_benchmark_results.txt)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        return 1
    fi
    
    # Cooling period after benchmark
    echo "System cooling after $algo..."
    sleep 45
    
    echo "" >> squaring_benchmark_results.txt
}

# List of squaring benchmark executables
SQUARING_ALGORITHMS=(
    "benchmark_squaring_newcol"
    "benchmark_squaring_newrow"
    "benchmark_squaring_ar24"
    "benchmark_squaring_jkls18"
    "benchmark_squaring_rt22"
    "benchmark_squaring_diag"
)

# Initialize result file
cat > squaring_benchmark_results.txt << EOL
Matrix Squaring Performance Benchmarks
-----------------------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------
EOL

# Run each benchmark
for algo in "${SQUARING_ALGORITHMS[@]}"; do
    run_benchmark $algo
done

# Add footer to results file
echo "-----------------------------------------------------------------------------------------------" >> squaring_benchmark_results.txt

echo "Squaring benchmarks complete. Results saved in squaring_benchmark_results.txt"