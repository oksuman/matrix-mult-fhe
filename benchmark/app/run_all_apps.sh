#!/bin/bash
# Sequential app benchmark runner: LDA → LR → FH(Diabetes)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$NCORES}

echo "======================================================================"
echo "  App Benchmark Suite: LDA / LR / FH(Diabetes)"
echo "  Start: $(date)"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "======================================================================"

# -------------------------------------------------------
# Step 1: LDA benchmark
# -------------------------------------------------------
echo ""
echo ">>> [1/3] LDA benchmark (start: $(date))"
bash "$SCRIPT_DIR/run_lda.sh"
echo ">>> LDA complete at $(date)"

echo ">>> Cooling 60s between apps..."
sleep 60

# -------------------------------------------------------
# Step 2: LR benchmark
# -------------------------------------------------------
echo ""
echo ">>> [2/3] LR benchmark (start: $(date))"
bash "$SCRIPT_DIR/run_lr.sh"
echo ">>> LR complete at $(date)"

echo ">>> Cooling 60s between apps..."
sleep 60

# -------------------------------------------------------
# Step 3: FH Diabetes benchmark
# -------------------------------------------------------
echo ""
echo ">>> [3/3] FH (Diabetes) benchmark (start: $(date))"
bash "$SCRIPT_DIR/run_fh.sh"
echo ">>> FH complete at $(date)"

# -------------------------------------------------------
# Done
# -------------------------------------------------------
echo ""
echo "======================================================================"
echo "  All benchmarks completed at $(date)"
echo "  Results:"
echo "    LDA : build/app/linear-discriminant-analysis/lda_results.txt"
echo "    LR  : build/app/linear-regression/lr_results.txt"
echo "    FH  : build/app/fixed-hessian/fh_results.txt"
echo "======================================================================"
