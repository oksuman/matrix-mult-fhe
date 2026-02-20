#!/bin/bash

# Generate summary tables for all application benchmarks
# Parses output from LR, LDA, and Fixed Hessian benchmarks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BASE="${SCRIPT_DIR}/../../build/app"

OUTPUT_FILE="app_benchmark_summary.txt"

cat > "$OUTPUT_FILE" << 'EOF'
================================================================================
              Application Benchmark Summary (NewCol vs AR24)
================================================================================

EOF

# ============ Linear Regression ============
echo "--- Linear Regression ---" >> "$OUTPUT_FILE"
LR_DIR="${BUILD_BASE}/linear-regression"

if [ -f "${LR_DIR}/newcol_timing.txt" ] && [ -f "${LR_DIR}/ar24_timing.txt" ]; then
    echo "" >> "$OUTPUT_FILE"
    printf "%-25s %15s %15s %15s\n" "Step" "NewCol (s)" "AR24 (s)" "Speedup" >> "$OUTPUT_FILE"
    echo "------------------------------------------------------------------------" >> "$OUTPUT_FILE"

    # Parse NewCol timings
    nc_step1=$(grep "Step 1" "${LR_DIR}/newcol_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    nc_step2=$(grep "Step 2" "${LR_DIR}/newcol_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    nc_step3=$(grep "Step 3" "${LR_DIR}/newcol_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    nc_step4=$(grep "Step 4" "${LR_DIR}/newcol_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    nc_total=$(grep "Total" "${LR_DIR}/newcol_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)

    # Parse AR24 timings
    ar_step1=$(grep "Step 1" "${LR_DIR}/ar24_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    ar_step2=$(grep "Step 2" "${LR_DIR}/ar24_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    ar_step3=$(grep "Step 3" "${LR_DIR}/ar24_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    ar_step4=$(grep "Step 4" "${LR_DIR}/ar24_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    ar_total=$(grep "Total" "${LR_DIR}/ar24_timing.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)

    # Print comparison
    if [ -n "$nc_step1" ] && [ -n "$ar_step1" ]; then
        speedup=$(echo "scale=2; $ar_step1 / $nc_step1" | bc 2>/dev/null || echo "N/A")
        printf "%-25s %15.2f %15.2f %15s\n" "X^T X" "$nc_step1" "$ar_step1" "${speedup}x" >> "$OUTPUT_FILE"
    fi
    if [ -n "$nc_step2" ] && [ -n "$ar_step2" ]; then
        speedup=$(echo "scale=2; $ar_step2 / $nc_step2" | bc 2>/dev/null || echo "N/A")
        printf "%-25s %15.2f %15.2f %15s\n" "Inverse" "$nc_step2" "$ar_step2" "${speedup}x" >> "$OUTPUT_FILE"
    fi
    if [ -n "$nc_step3" ] && [ -n "$ar_step3" ]; then
        speedup=$(echo "scale=2; $ar_step3 / $nc_step3" | bc 2>/dev/null || echo "N/A")
        printf "%-25s %15.2f %15.2f %15s\n" "X^T y" "$nc_step3" "$ar_step3" "${speedup}x" >> "$OUTPUT_FILE"
    fi
    if [ -n "$nc_step4" ] && [ -n "$ar_step4" ]; then
        speedup=$(echo "scale=2; $ar_step4 / $nc_step4" | bc 2>/dev/null || echo "N/A")
        printf "%-25s %15.2f %15.2f %15s\n" "Weights" "$nc_step4" "$ar_step4" "${speedup}x" >> "$OUTPUT_FILE"
    fi
    echo "------------------------------------------------------------------------" >> "$OUTPUT_FILE"
    if [ -n "$nc_total" ] && [ -n "$ar_total" ]; then
        speedup=$(echo "scale=2; $ar_total / $nc_total" | bc 2>/dev/null || echo "N/A")
        printf "%-25s %15.2f %15.2f %15s\n" "TOTAL" "$nc_total" "$ar_total" "${speedup}x" >> "$OUTPUT_FILE"
    fi

    # MSE comparison
    echo "" >> "$OUTPUT_FILE"
    echo "Accuracy (MSE):" >> "$OUTPUT_FILE"
    nc_mse=$(cat "${LR_DIR}/newcol_mse_result.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    ar_mse=$(cat "${LR_DIR}/ar24_mse_result.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
    printf "  NewCol: %s\n" "${nc_mse:-N/A}" >> "$OUTPUT_FILE"
    printf "  AR24:   %s\n" "${ar_mse:-N/A}" >> "$OUTPUT_FILE"
else
    echo "  (Results not found - run benchmark first)" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# ============ LDA ============
echo "--- Linear Discriminant Analysis ---" >> "$OUTPUT_FILE"
LDA_DIR="${BUILD_BASE}/linear-discriminant-analysis"

if [ -f "${LDA_DIR}/benchmark_results.txt" ]; then
    echo "" >> "$OUTPUT_FILE"
    # Extract timing from benchmark_results.txt
    grep -A 20 "TIMING COMPARISON" "${LDA_DIR}/benchmark_results.txt" 2>/dev/null | head -15 >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    grep -A 10 "ACCURACY" "${LDA_DIR}/benchmark_results.txt" 2>/dev/null | head -10 >> "$OUTPUT_FILE"
else
    echo "  (Results not found - run benchmark first)" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"

# ============ Fixed Hessian ============
echo "--- Fixed Hessian (Logistic Regression) ---" >> "$OUTPUT_FILE"
FH_DIR="${BUILD_BASE}/fixed-hessian"

echo "" >> "$OUTPUT_FILE"
echo "Note: Fixed Hessian results are printed to stdout." >> "$OUTPUT_FILE"
echo "Re-run with output redirection to capture results." >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=================================================================================" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"

echo "App benchmark summary generated: $OUTPUT_FILE"
