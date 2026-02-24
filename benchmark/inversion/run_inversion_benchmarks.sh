#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_inversion_benchmarks.sh
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

NUM_RUNS=${1:-1}
# 실험 모드: "original" / "simple" / "all" (기본: all)
BENCH_MODE=${2:-all}

echo "Running matrix inversion benchmarks..."
echo "Starting at $(date)"
echo "Runs per dimension: $NUM_RUNS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Benchmark mode: $BENCH_MODE"

cat > inversion_console.log << EOL
Matrix Inversion Performance Benchmarks
$(date)
Trials: $NUM_RUNS
OMP_NUM_THREADS: $OMP_NUM_THREADS
Benchmark mode: $BENCH_MODE
-----------------------------------------------------------------------------------------------
EOL

COOLING_PERIOD=45

run_benchmark() {
    local algo=$1
    local dim=$2
    echo ""
    echo "========================================"
    echo "  Running $algo d=$dim"
    echo "========================================"

    echo "System cooling (${COOLING_PERIOD}s)..."
    sleep $COOLING_PERIOD
    sync

    ./$algo $NUM_RUNS $dim 2>&1 | tee -a inversion_console.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error running $algo d=$dim"
        echo "[ERROR] $algo d=$dim failed" >> inversion_console.log
        return 1
    fi
}

# ============================================================
# Original inversion benchmarks (trace + eval_scalar_inverse)
# multDepth=31, levelBudget={4,5}
# ============================================================
if [ "$BENCH_MODE" = "original" ] || [ "$BENCH_MODE" = "all" ]; then
    echo ""
    echo "========================================"
    echo "  Original Inversion Benchmarks"
    echo "  (trace + eval_scalar_inverse)"
    echo "========================================"
    echo "[Original] benchmarks starting..." >> inversion_console.log

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_newcol" $dim
    done

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_ar24" $dim
    done

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_jkls18" $dim
    done

    # RT22: d=4,8,16,32 만 지원
    for dim in 4 8 16 32; do
        run_benchmark "benchmark_inversion_rt22" $dim
    done

    # Naive: d=4,8 만 지원
    for dim in 4 8; do
        run_benchmark "benchmark_inversion_naive" $dim
    done
fi

# ============================================================
# Simple inversion benchmarks (upperbound scaling, no trace)
# multDepth=30, levelBudget={4,4}, scale=1/d^2
# ============================================================
if [ "$BENCH_MODE" = "simple" ] || [ "$BENCH_MODE" = "all" ]; then
    echo ""
    echo "========================================"
    echo "  Simple Inversion Benchmarks"
    echo "  (1/d^2 upperbound, no trace)"
    echo "========================================"
    echo "[Simple] benchmarks starting..." >> inversion_console.log

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_simple_newcol" $dim
    done

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_simple_ar24" $dim
    done

    for dim in 4 8 16 32 64; do
        run_benchmark "benchmark_inversion_simple_jkls18" $dim
    done

    # RT22: d=4,8,16,32 만 지원
    for dim in 4 8 16 32; do
        run_benchmark "benchmark_inversion_simple_rt22" $dim
    done

    # Naive: d=4,8 만 지원
    for dim in 4 8; do
        run_benchmark "benchmark_inversion_naive" $dim
    done
fi

echo "Benchmarks completed at $(date)" >> inversion_console.log

echo ""
echo "Inversion benchmarks complete. Results saved in inversion_console.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_summary_table.sh" ]; then
    echo ""
    echo "Generating summary tables..."
    zsh "$SCRIPT_DIR/generate_summary_table.sh"
fi
