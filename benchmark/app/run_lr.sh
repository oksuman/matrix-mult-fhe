#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_lr.sh [NUM_RUNS]
#
# Usage:
#   ./run_lr.sh        # single run (original behavior)
#   ./run_lr.sh 10     # 10 repeated runs → aggregate into lr_results.txt
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

NUM_RUNS=${1:-1}
LOG_FILE="lr_console.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/linear-regression"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

echo "=============================================="
echo "  Linear Regression Benchmark"
[ "$NUM_RUNS" -gt 1 ] && echo "  Mode: $NUM_RUNS repeated runs"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Initialize console log
cat > "$LOG_FILE" << EOL
==============================================================================
  Linear Regression Benchmark Console Log
  Date: $(date)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
  Runs: $NUM_RUNS
==============================================================================

EOL

run_app() {
    local name=$1
    local exec=$2
    shift 2
    local args="$@"

    echo "----------------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------------"

    if [ -f "$exec" ]; then
        echo "Cooling down before $name..."
        sleep 30

        ./$exec $args 2>&1 | tee -a "$LOG_FILE"

        echo ""
        echo "Cooling down after $name..."
        sleep 30
    else
        echo "Warning: $exec not found, skipping..."
    fi
}

# Run plaintext baseline (deterministic — once regardless of NUM_RUNS)
run_app "Plaintext Baseline" "lr_plaintext"

# ============================================================
# Single run: original behavior
# ============================================================
if [ "$NUM_RUNS" -eq 1 ]; then
    run_app "Encrypted (Naive/NewCol/AR24)" "lr_benchmark" --benchmark

# ============================================================
# Multi-run: N times, save each result, then aggregate
# ============================================================
else
    RUNS_DIR="$BUILD_DIR/runs"
    mkdir -p "$RUNS_DIR"
    echo "Per-run results: $RUNS_DIR/"
    echo ""

    for i in $(seq 1 $NUM_RUNS); do
        RUN_NUM=$(printf "%02d" $i)

        echo ""
        echo "=============================================="
        echo "  Run $i / $NUM_RUNS  [$(date)]"
        echo "=============================================="
        echo "Cooling (45s)..."
        sleep 45
        sync

        echo "--- Run $i / $NUM_RUNS ---" >> "$LOG_FILE"
        ./lr_benchmark --benchmark 2>&1 | tee -a "$LOG_FILE" | tee "$RUNS_DIR/lr_console_run${RUN_NUM}.log"

        if [ -f "lr_results.txt" ]; then
            cp lr_results.txt "$RUNS_DIR/lr_results_run${RUN_NUM}.txt"
            echo "Saved: $RUNS_DIR/lr_results_run${RUN_NUM}.txt"
        else
            echo "Warning: lr_results.txt not found after run $i" >&2
        fi

        echo "Cooling (45s)..."
        sleep 45
    done

    echo ""
    echo "=============================================="
    echo "  All $NUM_RUNS runs done. Aggregating..."
    echo "=============================================="

    # Aggregate per-run results into final lr_results.txt using inline Python
    TMPPY=$(mktemp /tmp/aggregate_lr.XXXXXX.py)
    cat > "$TMPPY" << 'PYEOF'
import sys, re, math
from pathlib import Path
from datetime import datetime

runs_dir    = Path(sys.argv[1])
output_file = Path(sys.argv[2])
stats_file  = runs_dir / "lr_results_stats.txt"

def parse_lr(path):
    timing, mse = {}, {}
    text = Path(path).read_text()
    timing_m = re.search(r'TIMING COMPARISON.*?={3,}\n(.*?)={3,}', text, re.DOTALL)
    if timing_m:
        for line in timing_m.group(1).splitlines():
            m = re.match(r'\s*(Step\s+\d+|Total)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                timing[m.group(1).strip()] = {
                    'Naive': float(m.group(2)),
                    'AR24':  float(m.group(3)),
                    'NewCol':float(m.group(4)),
                }
    mse_m = re.search(r'ACCURACY.*?={3,}\n(.*?)={3,}', text, re.DOTALL)
    if mse_m:
        for line in mse_m.group(1).splitlines():
            m = re.match(r'\s*(Naive|AR24|NewCol)\s+([\d.]+)', line)
            if m:
                mse[m.group(1)] = float(m.group(2))
    return timing, mse

def stats(vals):
    n = len(vals)
    if n == 0: return 0.0, 0.0
    mu = sum(vals) / n
    sd = math.sqrt(sum((x - mu)**2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

files = sorted(runs_dir.glob("lr_results_run*.txt"))
if not files:
    print(f"Error: no lr_results_run*.txt in {runs_dir}"); sys.exit(1)

n = len(files)
print(f"  Aggregating {n} run(s)...")

all_timing, all_mse = [], []
for f in files:
    t, m = parse_lr(f)
    if t: all_timing.append(t)
    if m: all_mse.append(m)

steps = ['Step 1', 'Step 2', 'Step 3', 'Total']
algos = ['Naive', 'AR24', 'NewCol']

tstats = {}
for s in steps:
    tstats[s] = {}
    for a in algos:
        vals = [r[s][a] for r in all_timing if s in r and a in r.get(s, {})]
        tstats[s][a] = stats(vals)

mstats = {}
for a in algos:
    vals = [r[a] for r in all_mse if a in r]
    mstats[a] = stats(vals)

# Carry over non-timing sections from last run (deterministic)
last_text = files[-1].read_text()

def grab_section(text, header_pat):
    m = re.search(header_pat + r'.*?={3,}\n(.*?)(?=={3,}|\Z)', text, re.DOTALL)
    return m.group(1).rstrip('\n') if m else ""

config_sec = grab_section(last_text, r'--- Configuration ---')
comm_sec   = grab_section(last_text, r'COMMUNICATION COST')
mem_sec    = grab_section(last_text, r'MEMORY USAGE')

SEP  = "=" * 64
DASH = "-" * 70

lines = []
A = lines.append
A(SEP)
A(f"  LR Benchmark Results  (aggregated: {n} run{'s' if n > 1 else ''})")
A(f"  Generated: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
A(SEP)
A("")
A("--- Configuration ---")
A(config_sec)
A(f"  (Runs aggregated:     {n})")
A("")
A(SEP)
A("  TIMING COMPARISON (seconds)")
A(SEP)
A("")
A(f"{'Step':<20} {'Naive':>14} {'AR24':>14} {'NewCol':>14}")
A(DASH)
for s in steps:
    row = f"{s:<20}"
    for a in algos:
        mu, _ = tstats[s][a]
        row += f" {mu:>14.2f}"
    A(row)
if n > 1:
    A("")
    A("  Std Dev:")
    for s in steps:
        row = f"  {s:<18}"
        for a in algos:
            _, sd = tstats[s][a]
            row += f" {sd:>14.2f}"
        A(row)
A("")
A("  Step 1: Precomputation (X^T*X and X^T*y)")
A("  Step 2: Matrix Inversion")
A("  Step 3: Weight Computation")
A("")
A(SEP)
A("  ACCURACY (MSE)")
A(SEP)
A("")
A(f"{'Algorithm':<20} {'MSE':>12}")
A("-" * 34)
for a in algos:
    mu, sd = mstats[a]
    suf = f"  (±{sd:.6f})" if n > 1 and sd > 0 else ""
    A(f"{a:<20} {mu:>12.6f}{suf}")
A("")
if comm_sec:
    A(SEP); A("  COMMUNICATION COST (Serialized Sizes)"); A(SEP); A(""); A(comm_sec); A("")
if mem_sec:
    A(SEP); A("  MEMORY USAGE"); A(SEP); A(""); A(mem_sec); A("")
A(SEP); A("  END OF REPORT"); A(SEP)

output_file.write_text('\n'.join(lines) + '\n')
print(f"  -> {output_file}")

# Detailed stats file
stats_lines = [
    f"LR Benchmark Stats — {n} run(s)",
    f"Generated: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
    f"Files: {[f.name for f in files]}",
    "",
    "TIMING (mean ± stddev, seconds):",
    f"{'Step':<20} {'Naive':>22} {'AR24':>22} {'NewCol':>22}",
    "-" * 90,
]
for s in steps:
    row = f"{s:<20}"
    for a in algos:
        mu, sd = tstats[s][a]
        cell = f"{mu:.2f}±{sd:.2f}" if n > 1 else f"{mu:.2f}"
        row += f" {cell:>22}"
    stats_lines.append(row)
stats_lines += ["", "MSE (mean ± stddev):"]
for a in algos:
    mu, sd = mstats[a]
    cell = f"{mu:.6f}±{sd:.6f}" if n > 1 else f"{mu:.6f}"
    stats_lines.append(f"  {a:<10} {cell}")

stats_file.write_text('\n'.join(stats_lines) + '\n')
print(f"  -> {stats_file} (detailed stats)")
PYEOF

    python3 "$TMPPY" "$RUNS_DIR" "$BUILD_DIR/lr_results.txt"
    rm -f "$TMPPY"
fi

echo ""
echo "=============================================="
echo "  Linear Regression Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Console log: $BUILD_DIR/$LOG_FILE"
if [ "$NUM_RUNS" -gt 1 ]; then
    echo "Per-run results: $BUILD_DIR/runs/"
    echo "Stats detail:   $BUILD_DIR/runs/lr_results_stats.txt"
fi
echo "Result file:    $BUILD_DIR/lr_results.txt"
