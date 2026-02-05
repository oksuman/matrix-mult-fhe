#!/bin/bash

echo "Running matrix inversion benchmarks..."
echo "Starting at $(date)"

# Initialize result file with header
cat > inversion_benchmark_results.txt << EOL
Matrix Inversion Performance Benchmarks
$(date)
-----------------------------------------------------------------------------------------------
EOL

run_benchmark() {
    local algo=$1
    echo "Running $algo..."
    
    # Cooling period before benchmark
    echo "System cooling before $algo..."
    sleep 45
    sync
    
    # Create temporary file for benchmark output
    temp_output=$(mktemp)
    
    # Run benchmark and save complete output
    ./$algo | tee "$temp_output"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo"
        rm "$temp_output"
        return 1
    fi
    
    # Extract and write ring dimension information
    echo "Ring Dimensions for $algo:" >> inversion_benchmark_results.txt
    grep "Ring Dimension:" "$temp_output" >> inversion_benchmark_results.txt
    echo "" >> inversion_benchmark_results.txt
    
    # Write benchmark results
    echo "Benchmark Results:" >> inversion_benchmark_results.txt
    grep "BM_" "$temp_output" | grep -v "^Running" >> inversion_benchmark_results.txt
    echo "-----------------------------------------------------------------------------------------------" >> inversion_benchmark_results.txt
    echo "" >> inversion_benchmark_results.txt
    
    # Clean up temporary file
    rm "$temp_output"
    
    # Cooling period after benchmark
    echo "System cooling after $algo..."
    sleep 45
}

# List of inversion benchmark executables
INVERSION_ALGORITHMS=(
    "benchmark_inversion_newcol"
    "benchmark_inversion_newrow"
    "benchmark_inversion_ar24"
    "benchmark_inversion_jkls18"
    "benchmark_inversion_rt22"
    "benchmark_inversion_diag"
    "benchmark_inversion_naive"
)

# Run each benchmark
for algo in "${INVERSION_ALGORITHMS[@]}"; do
    run_benchmark $algo
done

# Add footer to results file
echo "Benchmarks completed at $(date)" >> inversion_benchmark_results.txt

echo "Inversion benchmarks complete. Results saved in inversion_benchmark_results.txt"