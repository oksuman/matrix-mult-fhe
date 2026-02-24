#!/bin/bash
# Combines lda_results.txt, lr_results.txt, fh_results.txt into one summary file.
# Run from project root: bash benchmark/app/generate_app_summary.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BASE="${SCRIPT_DIR}/../../build/app"

LDA_FILE="${BUILD_BASE}/linear-discriminant-analysis/lda_results.txt"
LR_FILE="${BUILD_BASE}/linear-regression/lr_results.txt"
FH_FILE="${BUILD_BASE}/fixed-hessian/fh_results.txt"

OUTPUT_FILE="${BUILD_BASE}/app_summary.txt"

{
echo "================================================================================"
echo "  Application Benchmark Summary"
echo "  Generated: $(date)"
echo "================================================================================"
echo ""

echo "################################################################################"
echo "  [1/3] Linear Discriminant Analysis (LDA)"
echo "################################################################################"
if [ -f "$LDA_FILE" ]; then
    cat "$LDA_FILE"
else
    echo "  (not found: $LDA_FILE)"
fi

echo ""
echo "################################################################################"
echo "  [2/3] Linear Regression (LR)"
echo "################################################################################"
if [ -f "$LR_FILE" ]; then
    cat "$LR_FILE"
else
    echo "  (not found: $LR_FILE)"
fi

echo ""
echo "################################################################################"
echo "  [3/3] Fixed Hessian (FH)"
echo "################################################################################"
if [ -f "$FH_FILE" ]; then
    cat "$FH_FILE"
else
    echo "  (not found: $FH_FILE)"
fi

echo ""
echo "================================================================================"
echo "  END"
echo "================================================================================"
} > "$OUTPUT_FILE"

echo "Summary saved to: $OUTPUT_FILE"
