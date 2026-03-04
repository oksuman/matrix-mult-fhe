#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_lda.sh [NUM_RUNS]
#
# Usage:
#   ./run_lda.sh        # single run (original behavior)
#   ./run_lda.sh 10     # 10 repeated runs → aggregate into lda_results.txt
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

NUM_RUNS=${1:-1}
LOG_FILE="lda_console.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/linear-discriminant-analysis"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

echo "=============================================="
echo "  LDA Benchmark"
[ "$NUM_RUNS" -gt 1 ] && echo "  Mode: $NUM_RUNS repeated runs"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Initialize console log
cat > "$LOG_FILE" << EOL
==============================================================================
  LDA Benchmark Console Log
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
run_app "Plaintext Baseline" "lda_plaintext" --no-save

# ============================================================
# Single run: original behavior
# ============================================================
if [ "$NUM_RUNS" -eq 1 ]; then
    run_app "Benchmark LDA" "lda_benchmark"

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
        ./lda_benchmark 2>&1 | tee -a "$LOG_FILE" | tee "$RUNS_DIR/lda_console_run${RUN_NUM}.log"

        if [ -f "lda_results.txt" ]; then
            cp lda_results.txt "$RUNS_DIR/lda_results_run${RUN_NUM}.txt"
            echo "Saved: $RUNS_DIR/lda_results_run${RUN_NUM}.txt"
        else
            echo "Warning: lda_results.txt not found after run $i" >&2
        fi

        echo "Cooling (45s)..."
        sleep 45
    done

    echo ""
    echo "=============================================="
    echo "  All $NUM_RUNS runs done. Aggregating..."
    echo "=============================================="

    TMPPY=$(mktemp /tmp/aggregate_lda.XXXXXX.py)
    cat > "$TMPPY" << 'PYEOF'
import sys, re, math
from pathlib import Path
from datetime import datetime

runs_dir    = Path(sys.argv[1])
output_file = Path(sys.argv[2])
stats_file  = runs_dir / "lda_results_stats.txt"

def parse_lda(path):
    # Bug fix: skip the === separator that immediately follows the TIMING COMPARISON header
    timing = {}
    in_timing = False
    passed_first_sep = False
    for line in Path(path).read_text().splitlines():
        if 'TIMING COMPARISON' in line:
            in_timing = True
            passed_first_sep = False
            continue
        if re.match(r'={3,}', line):
            if in_timing and not passed_first_sep:
                passed_first_sep = True
                continue
            in_timing = False
        if in_timing:
            m = re.match(r'\s+(.+?)\s{2,}(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', line)
            if m:
                step = m.group(1).strip()
                timing[step] = {
                    'NewCol': float(m.group(2)),
                    'AR24':   float(m.group(3)),
                }
    return timing

def stats(vals):
    n = len(vals)
    if n == 0: return 0.0, 0.0
    mu = sum(vals) / n
    sd = math.sqrt(sum((x - mu)**2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

files = sorted(runs_dir.glob("lda_results_run*.txt"))
if not files:
    print(f"Error: no lda_results_run*.txt in {runs_dir}"); sys.exit(1)

n = len(files)
print(f"  Aggregating {n} run(s)...")

all_timing = []
for f in files:
    t = parse_lda(f)
    if t: all_timing.append(t)

steps = ['S_W computation', 'S_W^{-1} computation', 'w computation', 'TOTAL']
algos = ['NewCol', 'AR24']

tstats = {}
for s in steps:
    tstats[s] = {}
    for a in algos:
        vals = [r[s][a] for r in all_timing if s in r and a in r.get(s, {})]
        tstats[s][a] = stats(vals)

# Carry over non-timing sections from last run
last_text = files[-1].read_text()

def grab_config(text):
    # Format: --- Configuration ---\n<content>\n===...
    lines = text.splitlines()
    in_section, content = False, []
    for line in lines:
        if '--- Configuration ---' in line:
            in_section = True; continue
        if in_section:
            if re.match(r'={3,}', line): break
            content.append(line)
    return '\n'.join(content).rstrip('\n')

def grab_section_with_sep(text, keyword):
    # Format: ===\n  KEYWORD\n===\n<content>\n===
    lines = text.splitlines()
    in_section, passed_sep, content = False, False, []
    for line in lines:
        if keyword in line and not in_section:
            in_section = True; passed_sep = False; content = []; continue
        if in_section:
            if not passed_sep and re.match(r'={3,}', line):
                passed_sep = True; continue
            if passed_sep:
                if re.match(r'={3,}', line): break
                content.append(line)
    return '\n'.join(content).rstrip('\n')

config_sec    = grab_config(last_text)
accuracy_sec  = grab_section_with_sep(last_text, 'ACCURACY & F1 SCORE')
confusion_sec = grab_section_with_sep(last_text, 'CONFUSION MATRICES')

SEP  = "=" * 64
DASH = "-" * 70

lines = []
A = lines.append
A(SEP)
A(f"  LDA Benchmark Results  (aggregated: {n} run{'s' if n > 1 else ''})")
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
A(f"{'Step':>30} {'NewCol':>14} {'AR24':>14} {'Diff':>14}")
A(DASH)
for s in steps:
    nc_mu, _ = tstats[s]['NewCol']
    ar_mu, _ = tstats[s]['AR24']
    diff = nc_mu - ar_mu
    A(f"{s:>30} {nc_mu:>14.4f} {ar_mu:>14.4f} {diff:>14.4f}")
if n > 1:
    A("")
    A("  Std Dev:")
    for s in steps:
        _, nc_sd = tstats[s]['NewCol']
        _, ar_sd = tstats[s]['AR24']
        A(f"  {s:>28} {nc_sd:>14.4f} {ar_sd:>14.4f}")
A("")
if accuracy_sec:
    A(SEP); A("  ACCURACY & F1 SCORE"); A(SEP); A(""); A(accuracy_sec); A("")
if confusion_sec:
    A(SEP); A("  CONFUSION MATRICES"); A(SEP); A(""); A(confusion_sec); A("")
A(SEP); A("  END OF REPORT"); A(SEP)

output_file.write_text('\n'.join(lines) + '\n')
print(f"  -> {output_file}")

# Stats file
stats_lines = [
    f"LDA Benchmark Stats — {n} run(s)",
    f"Generated: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
    f"Files: {[f.name for f in files]}",
    "",
    "TIMING (mean ± stddev, seconds):",
    f"{'Step':>30} {'NewCol':>22} {'AR24':>22}",
    "-" * 78,
]
for s in steps:
    nc_mu, nc_sd = tstats[s]['NewCol']
    ar_mu, ar_sd = tstats[s]['AR24']
    nc_cell = f"{nc_mu:.4f}±{nc_sd:.4f}" if n > 1 else f"{nc_mu:.4f}"
    ar_cell = f"{ar_mu:.4f}±{ar_sd:.4f}" if n > 1 else f"{ar_mu:.4f}"
    stats_lines.append(f"{s:>30} {nc_cell:>22} {ar_cell:>22}")

stats_file.write_text('\n'.join(stats_lines) + '\n')
print(f"  -> {stats_file} (detailed stats)")
PYEOF

    python3 "$TMPPY" "$RUNS_DIR" "$BUILD_DIR/lda_results.txt"
    rm -f "$TMPPY"
fi

echo ""
echo "=============================================="
echo "  LDA Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Console log: $BUILD_DIR/$LOG_FILE"
if [ "$NUM_RUNS" -gt 1 ]; then
    echo "Per-run results: $BUILD_DIR/runs/"
    echo "Stats detail:   $BUILD_DIR/runs/lda_results_stats.txt"
fi
echo "Result file:    $BUILD_DIR/lda_results.txt"
