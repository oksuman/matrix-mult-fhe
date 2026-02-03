#!/bin/bash

echo "Running matrix multiplication benchmarks..."
sleep 2

run_benchmark() {
    local algo=$1
    echo "Running $algo..."
    
    # Add cooling period
    echo "System cooling before $algo..."
    sleep 45
    sync
    
    # Run benchmark with direct console output
    ./$algo | tee >(grep "BM_" | grep -v "^Running" >> benchmark_results.txt)
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        return 1
    fi
    
    # Additional cooling period
    echo "System cooling after $algo..."
    sleep 45
    
    echo "" >> benchmark_results.txt
}

ALGORITHMS=(
    "benchmark_diag"
    # "benchmark_newcol" 
    # "benchmark_newrow" 
    # "benchmark_as24" 
    # "benchmark_jkls18" 
    # "benchmark_rt22" 
)

# Initialize result file with header
cat > benchmark_results.txt << EOL
Matrix Multiplication Performance
-----------------------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations UserCounters...
-----------------------------------------------------------------------------------------------
EOL

for algo in "${ALGORITHMS[@]}"; do
    run_benchmark $algo
done

# Add footer to file
echo "-----------------------------------------------------------------------------------------------" >> benchmark_results.txt

echo "Benchmarking complete. Results saved in benchmark_results.txt"