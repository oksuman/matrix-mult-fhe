#!/bin/zsh

# Parse squaring benchmark results and generate summary tables

INPUT_FILE="squaring_benchmark_results.txt"
TIME_TABLE="squaring_summary_time.txt"
MEMORY_TABLE="squaring_summary_memory.txt"
ACCURACY_TABLE="squaring_summary_accuracy.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found"
    exit 1
fi

# Use typeset -A for zsh associative arrays
typeset -A times
typeset -A accuracies
typeset -A setup_mem
typeset -A peak_mem
typeset -A ct_size
typeset -A ct_count
typeset -A rot_key_size
typeset -A relin_key_size

# Parse the results file
current_algo=""
current_dim=""

while IFS= read -r line; do
    # Detect algorithm name
    if [[ $line =~ "Matrix Squaring Benchmark - "([A-Za-z0-9]+) ]]; then
        current_algo="${match[1]}"
    fi

    # Detect dimension
    if [[ $line =~ "========== "([A-Za-z0-9]+)" d="([0-9]+) ]]; then
        current_algo="${match[1]}"
        current_dim="${match[2]}"
    fi

    # Parse time from summary
    if [[ $line =~ "Time: "([0-9.]+)"s" ]]; then
        times[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse log2 error
    if [[ $line =~ "log2\(Rel\. Frob\.\):"[[:space:]]*(-?[0-9.]+) ]]; then
        accuracies[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse memory metrics
    if [[ $line =~ "Setup Memory:"[[:space:]]*([0-9.]+)" GB" ]]; then
        setup_mem[${current_algo}_${current_dim}]="${match[1]}"
    fi

    if [[ $line =~ "Peak Memory:"[[:space:]]*([0-9.]+)" GB" ]]; then
        peak_mem[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse ciphertext info
    if [[ $line =~ "Ciphertext:"[[:space:]]*([0-9.]+)" MB" ]]; then
        ct_size[${current_algo}_${current_dim}]="${match[1]}"
    fi

    if [[ $line =~ "x "([0-9]+)" = " ]]; then
        ct_count[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse key sizes
    if [[ $line =~ "Rotation Keys:"[[:space:]]*([0-9.]+)" MB" ]]; then
        rot_key_size[${current_algo}_${current_dim}]="${match[1]}"
    fi

    if [[ $line =~ "Relin Key:"[[:space:]]*([0-9.]+)" MB" ]]; then
        relin_key_size[${current_algo}_${current_dim}]="${match[1]}"
    fi

done < "$INPUT_FILE"

# Get list of algorithms and dimensions
ALGORITHMS=(Naive NewCol AR24 JKLS18 RT22)
DIMENSIONS=(4 8 16 32 64)

# Generate Time Table
cat > "$TIME_TABLE" << 'EOF'
================================================================================
                    Matrix Squaring Time Comparison (seconds)
================================================================================
EOF

printf "%-12s" "Algorithm" >> "$TIME_TABLE"
for d in "${DIMENSIONS[@]}"; do
    printf "%12s" "d=$d" >> "$TIME_TABLE"
done
echo "" >> "$TIME_TABLE"
echo "--------------------------------------------------------------------------------" >> "$TIME_TABLE"

for algo in "${ALGORITHMS[@]}"; do
    printf "%-12s" "$algo" >> "$TIME_TABLE"
    for d in "${DIMENSIONS[@]}"; do
        key="${algo}_${d}"
        if [ -n "${times[$key]}" ]; then
            printf "%12.2f" "${times[$key]}" >> "$TIME_TABLE"
        else
            printf "%12s" "-" >> "$TIME_TABLE"
        fi
    done
    echo "" >> "$TIME_TABLE"
done

echo "--------------------------------------------------------------------------------" >> "$TIME_TABLE"
echo "Generated: $(date)" >> "$TIME_TABLE"

# Generate Memory Table
cat > "$MEMORY_TABLE" << 'EOF'
================================================================================
                    Matrix Squaring Memory Comparison
================================================================================
  Setup Overhead = Setup - Idle
  Runtime Overhead = Peak - Setup
--------------------------------------------------------------------------------
EOF

printf "%-10s %4s" "Algorithm" "d" >> "$MEMORY_TABLE"
printf "%12s" "Setup OH" >> "$MEMORY_TABLE"
printf "%12s" "Runtime OH" >> "$MEMORY_TABLE"
printf "%12s" "CT(MB)" >> "$MEMORY_TABLE"
printf "%10s" "CT Cnt" >> "$MEMORY_TABLE"
printf "%12s" "RotKey(MB)" >> "$MEMORY_TABLE"
printf "%12s" "Relin(MB)" >> "$MEMORY_TABLE"
echo "" >> "$MEMORY_TABLE"
echo "--------------------------------------------------------------------------------" >> "$MEMORY_TABLE"

for algo in "${ALGORITHMS[@]}"; do
    for d in "${DIMENSIONS[@]}"; do
        key="${algo}_${d}"
        if [ -n "${times[$key]}" ]; then
            printf "%-10s %4d" "$algo" "$d" >> "$MEMORY_TABLE"

            setup="${setup_mem[$key]:-0}"
            peak="${peak_mem[$key]:-0}"
            setup_oh=$(echo "scale=2; $setup - 0.004" | bc 2>/dev/null || echo "$setup")
            runtime_oh=$(echo "scale=2; $peak - $setup" | bc 2>/dev/null || echo "0")

            printf "%12s" "$setup_oh" >> "$MEMORY_TABLE"
            printf "%12s" "$runtime_oh" >> "$MEMORY_TABLE"
            printf "%12s" "${ct_size[$key]:-N/A}" >> "$MEMORY_TABLE"
            printf "%10s" "${ct_count[$key]:-1}" >> "$MEMORY_TABLE"
            printf "%12s" "${rot_key_size[$key]:-0}" >> "$MEMORY_TABLE"
            printf "%12s" "${relin_key_size[$key]:-N/A}" >> "$MEMORY_TABLE"
            echo "" >> "$MEMORY_TABLE"
        fi
    done
done

echo "--------------------------------------------------------------------------------" >> "$MEMORY_TABLE"
echo "Generated: $(date)" >> "$MEMORY_TABLE"

# Generate Accuracy Table
cat > "$ACCURACY_TABLE" << 'EOF'
================================================================================
                    Matrix Squaring Accuracy Comparison (log2 error)
================================================================================
EOF

printf "%-12s" "Algorithm" >> "$ACCURACY_TABLE"
for d in "${DIMENSIONS[@]}"; do
    printf "%12s" "d=$d" >> "$ACCURACY_TABLE"
done
echo "" >> "$ACCURACY_TABLE"
echo "--------------------------------------------------------------------------------" >> "$ACCURACY_TABLE"

for algo in "${ALGORITHMS[@]}"; do
    printf "%-12s" "$algo" >> "$ACCURACY_TABLE"
    for d in "${DIMENSIONS[@]}"; do
        key="${algo}_${d}"
        if [ -n "${accuracies[$key]}" ]; then
            printf "%12.1f" "${accuracies[$key]}" >> "$ACCURACY_TABLE"
        else
            printf "%12s" "-" >> "$ACCURACY_TABLE"
        fi
    done
    echo "" >> "$ACCURACY_TABLE"
done

echo "--------------------------------------------------------------------------------" >> "$ACCURACY_TABLE"
echo "Generated: $(date)" >> "$ACCURACY_TABLE"

echo "Summary tables generated:"
echo "  - $TIME_TABLE"
echo "  - $MEMORY_TABLE"
echo "  - $ACCURACY_TABLE"
