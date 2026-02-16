#!/bin/zsh

# Parse inversion benchmark results and generate summary tables

INPUT_FILE="inversion_benchmark_results.txt"
TIME_TABLE="inversion_summary_time.txt"
ACCURACY_TABLE="inversion_summary_accuracy.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found"
    exit 1
fi

# Use typeset -A for zsh associative arrays
typeset -A times
typeset -A log2_frob
typeset -A log2_max

# Parse the results file
current_algo=""
current_dim=""

while IFS= read -r line; do
    # Detect algorithm from header
    if [[ $line =~ "Matrix Inversion Benchmark - "([A-Za-z0-9]+) ]]; then
        current_algo="${match[1]}"
    fi

    # Detect dimension from summary header
    if [[ $line =~ "========== "([A-Za-z0-9]+)" Inversion d="([0-9]+) ]]; then
        current_algo="${match[1]}"
        current_dim="${match[2]}"
    fi

    # Parse time from summary
    if [[ $line =~ "Time: "([0-9.]+)"s" ]]; then
        times[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse log2 Frobenius error
    if [[ $line =~ "log2\(Rel\. Frob\.\):"[[:space:]]*(-?[0-9.]+) ]]; then
        log2_frob[${current_algo}_${current_dim}]="${match[1]}"
    fi

    # Parse log2 Max error
    if [[ $line =~ "log2\(Rel\. Max\):"[[:space:]]*(-?[0-9.]+) ]]; then
        log2_max[${current_algo}_${current_dim}]="${match[1]}"
    fi

done < "$INPUT_FILE"

# Algorithm and dimension lists
ALGORITHMS=(Naive NewCol AR24 JKLS18 RT22)
DIMENSIONS=(4 8 16 32 64)

# Generate Time Comparison Table
cat > "$TIME_TABLE" << 'EOF'
================================================================================
                    Matrix Inversion Time Comparison (seconds)
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

# Generate Accuracy Comparison Table
cat > "$ACCURACY_TABLE" << 'EOF'
================================================================================
                    Matrix Inversion Accuracy Comparison
================================================================================
                              log2(Relative Frobenius Error)
--------------------------------------------------------------------------------
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
        if [ -n "${log2_frob[$key]}" ]; then
            printf "%12.1f" "${log2_frob[$key]}" >> "$ACCURACY_TABLE"
        else
            printf "%12s" "-" >> "$ACCURACY_TABLE"
        fi
    done
    echo "" >> "$ACCURACY_TABLE"
done

echo "" >> "$ACCURACY_TABLE"
echo "                              log2(Relative Max Error)" >> "$ACCURACY_TABLE"
echo "--------------------------------------------------------------------------------" >> "$ACCURACY_TABLE"

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
        if [ -n "${log2_max[$key]}" ]; then
            printf "%12.1f" "${log2_max[$key]}" >> "$ACCURACY_TABLE"
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
echo "  - $ACCURACY_TABLE"
