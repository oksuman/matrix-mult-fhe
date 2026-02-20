#!/bin/bash

# =============================================================
# Matrix-Mult-FHE: Unified Benchmark Runner
# =============================================================
# This script runs all benchmarks sequentially:
# 1. Matrix Squaring
# 2. Matrix Inversion
# 3. Linear Regression
# 4. LDA
# 5. Fixed Hessian
#
# Results are saved to the results/ directory.
# =============================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure single-thread execution
export OMP_NUM_THREADS=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR"

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this run
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}  Matrix-Mult-FHE Benchmark Suite${NC}"
echo -e "${BLUE}=============================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS (Single Thread Mode)"
echo "  Results directory: $RESULTS_DIR"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Function to run a benchmark module
run_benchmark() {
    local name=$1
    local script=$2
    local output_file=$3

    echo -e "${YELLOW}----------------------------------------------------------${NC}"
    echo -e "${YELLOW}Running: $name${NC}"
    echo -e "${YELLOW}----------------------------------------------------------${NC}"

    if [ -f "$script" ]; then
        chmod +x "$script"

        # Run benchmark and capture output
        echo "Start time: $(date)"
        "$script" 2>&1 | tee "$output_file"

        echo "End time: $(date)"
        echo -e "${GREEN}✓ $name completed. Results saved to $output_file${NC}"
    else
        echo -e "${RED}✗ Script not found: $script${NC}"
        return 1
    fi

    # Cooling period between benchmarks
    echo "Cooling down for 60 seconds..."
    sleep 60
}

# Function to print summary
print_summary() {
    echo ""
    echo -e "${BLUE}=============================================================${NC}"
    echo -e "${BLUE}  Benchmark Summary${NC}"
    echo -e "${BLUE}=============================================================${NC}"
    echo ""
    echo "Results saved in: $RESULTS_DIR"
    echo ""
    ls -la "$RESULTS_DIR"/*.txt 2>/dev/null || echo "No result files found."
}

# Main execution
main() {
    local start_time=$(date +%s)

    # Check which benchmarks to run (default: all)
    local run_squaring=true
    local run_inversion=true
    local run_apps=true

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --squaring-only)
                run_inversion=false
                run_apps=false
                shift
                ;;
            --inversion-only)
                run_squaring=false
                run_apps=false
                shift
                ;;
            --apps-only)
                run_squaring=false
                run_inversion=false
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --squaring-only    Run only matrix squaring benchmarks"
                echo "  --inversion-only   Run only inversion benchmarks"
                echo "  --apps-only        Run only application benchmarks"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done

    # 1. Matrix Squaring
    if $run_squaring; then
        if [ -d "$BUILD_DIR/squaring" ]; then
            run_benchmark "Matrix Squaring" \
                "$BUILD_DIR/squaring/run_squaring_benchmarks.sh" \
                "$RESULTS_DIR/squaring_results_$TIMESTAMP.txt"
        else
            echo -e "${RED}Squaring directory not found${NC}"
        fi
    fi

    # 2. Matrix Inversion
    if $run_inversion; then
        if [ -d "$BUILD_DIR/inversion" ]; then
            run_benchmark "Matrix Inversion" \
                "$BUILD_DIR/inversion/run_inversion_benchmarks.sh" \
                "$RESULTS_DIR/inversion_results_$TIMESTAMP.txt"
        else
            echo -e "${RED}Inversion directory not found${NC}"
        fi
    fi

    # 3. Applications (use benchmark/app scripts)
    if $run_apps; then
        local APP_SCRIPT_DIR="$SCRIPT_DIR/app"
        if [ -d "$APP_SCRIPT_DIR" ]; then
            # Linear Regression
            run_benchmark "Linear Regression" \
                "$APP_SCRIPT_DIR/run_lr.sh" \
                "$RESULTS_DIR/lr_results_$TIMESTAMP.txt"

            # LDA
            run_benchmark "Linear Discriminant Analysis" \
                "$APP_SCRIPT_DIR/run_lda.sh" \
                "$RESULTS_DIR/lda_results_$TIMESTAMP.txt"

            # Fixed Hessian
            run_benchmark "Fixed Hessian" \
                "$APP_SCRIPT_DIR/run_fh.sh" \
                "$RESULTS_DIR/fh_results_$TIMESTAMP.txt"
        else
            echo -e "${YELLOW}Applications directory not found. Skipping app benchmarks.${NC}"
        fi
    fi

    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))

    # Print summary
    print_summary

    echo ""
    echo -e "${GREEN}Total execution time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${GREEN}All benchmarks completed!${NC}"
}

# Run main function
main "$@"
