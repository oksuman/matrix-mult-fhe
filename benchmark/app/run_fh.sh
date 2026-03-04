#!/bin/bash

# ============================================================
# Hardware Setting (shared across all benchmark scripts)
# Override: OMP_NUM_THREADS=8 ./run_fh.sh [NUM_RUNS]
#
# Usage:
#   ./run_fh.sh        # single run (original behavior)
#   ./run_fh.sh 10     # 10 repeated runs → aggregate into fh_results.txt
# ============================================================
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc --all 2>/dev/null || echo 16)
    export OMP_NUM_THREADS=$NCORES
fi

NUM_RUNS=${1:-1}
LOG_FILE="fh_console.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build/app/fixed-hessian"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

echo "=============================================="
echo "  Fixed Hessian Benchmark"
[ "$NUM_RUNS" -gt 1 ] && echo "  Mode: $NUM_RUNS repeated runs"
echo "=============================================="
echo "Start time: $(date)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Initialize console log
cat > "$LOG_FILE" << EOL
==============================================================================
  Fixed Hessian Benchmark Console Log
  Date: $(date)
  OMP_NUM_THREADS: $OMP_NUM_THREADS
  Dataset: Diabetes (64 train, 256 test)
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
run_app "Plaintext (Diabetes)" "fh_plaintext_diabetes"

# ============================================================
# Single run: original behavior
# ============================================================
if [ "$NUM_RUNS" -eq 1 ]; then
    run_app "Encrypted (Diabetes)" "fh_encrypted_diabetes" --benchmark

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
        ./fh_encrypted_diabetes --benchmark 2>&1 | tee -a "$LOG_FILE" | tee "$RUNS_DIR/fh_console_run${RUN_NUM}.log"

        if [ -f "fh_results.txt" ]; then
            cp fh_results.txt "$RUNS_DIR/fh_results_run${RUN_NUM}.txt"
            echo "Saved: $RUNS_DIR/fh_results_run${RUN_NUM}.txt"
        else
            echo "Warning: fh_results.txt not found after run $i" >&2
        fi

        echo "Cooling (45s)..."
        sleep 45
    done

    echo ""
    echo "=============================================="
    echo "  All $NUM_RUNS runs done. Aggregating..."
    echo "=============================================="

    TMPPY=$(mktemp /tmp/aggregate_fh.XXXXXX.py)
    cat > "$TMPPY" << 'PYEOF'
import sys, re, math
from pathlib import Path
from datetime import datetime

runs_dir    = Path(sys.argv[1])
output_file = Path(sys.argv[2])
stats_file  = runs_dir / "fh_results_stats.txt"

def parse_fh(path):
    # Parse table rows: "  FH (AR24)  |  1  |  460.58  |  80.08%  |  67.92%"
    results = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(
            r'\s+(FH \(AR24\)|FH \(NewCol\)|SFH)\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%',
            line
        )
        if m:
            method = m.group(1)
            iters  = int(m.group(2))
            key    = (method, iters)
            results[key] = {
                'time':     float(m.group(3)),
                'accuracy': float(m.group(4)),
                'f1':       float(m.group(5)),
            }
    return results

def stats(vals):
    n = len(vals)
    if n == 0: return 0.0, 0.0
    mu = sum(vals) / n
    sd = math.sqrt(sum((x - mu)**2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

files = sorted(runs_dir.glob("fh_results_run*.txt"))
if not files:
    print(f"Error: no fh_results_run*.txt in {runs_dir}"); sys.exit(1)

n = len(files)
print(f"  Aggregating {n} run(s)...")

all_results = []
for f in files:
    r = parse_fh(f)
    if r: all_results.append(r)

# Collect all keys (preserve order from first file)
keys = list(all_results[0].keys()) if all_results else []

tstats = {}
for key in keys:
    times     = [r[key]['time']     for r in all_results if key in r]
    accuracies= [r[key]['accuracy'] for r in all_results if key in r]
    f1s       = [r[key]['f1']       for r in all_results if key in r]
    tstats[key] = {
        'time':     stats(times),
        'accuracy': stats(accuracies),
        'f1':       stats(f1s),
    }

# Carry over config from last run
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

config_sec = grab_config(last_text)

SEP  = "=" * 64
DASH = "-" * 62

lines = []
A = lines.append
A(SEP)
A(f"  FH Benchmark Results  (aggregated: {n} run{'s' if n > 1 else ''})")
A(f"  Generated: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
A(SEP)
A("")
A("--- Configuration ---")
A(config_sec)
A(f"  (Runs aggregated:     {n})")
A("")
A(SEP)
A("  TIMING & ACCURACY COMPARISON")
A(SEP)
A("")
A(f"  {'Method':<20} | {'Iter':>4} | {'Time (s)':>9} | {'Accuracy':>9} | {'F1 Score':>8}")
A(f"  {'-'*60}")
for key in keys:
    method, iters = key
    t_mu,  t_sd  = tstats[key]['time']
    a_mu,  _     = tstats[key]['accuracy']
    f1_mu, _     = tstats[key]['f1']
    A(f"  {method:<20} | {iters:>4} | {t_mu:>9.2f} | {a_mu:>8.2f}% | {f1_mu:>7.2f}%")
if n > 1:
    A(f"  {'='*60}")
    A("  Std Dev (time):")
    for key in keys:
        method, iters = key
        _, t_sd = tstats[key]['time']
        A(f"  {method:<20} | {iters:>4} | {t_sd:>9.2f}")
A(f"  {'='*60}")
A("")
A(SEP); A("  END OF REPORT"); A(SEP)

output_file.write_text('\n'.join(lines) + '\n')
print(f"  -> {output_file}")

# Stats file
stats_lines = [
    f"FH Benchmark Stats — {n} run(s)",
    f"Generated: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
    f"Files: {[f.name for f in files]}",
    "",
    f"{'Method':<20} {'Iter':>6} {'Time mean±std':>20} {'Accuracy':>12} {'F1':>12}",
    "-" * 74,
]
for key in keys:
    method, iters = key
    t_mu, t_sd = tstats[key]['time']
    a_mu, _    = tstats[key]['accuracy']
    f1_mu, _   = tstats[key]['f1']
    cell = f"{t_mu:.2f}±{t_sd:.2f}" if n > 1 else f"{t_mu:.2f}"
    stats_lines.append(f"{method:<20} {iters:>6} {cell:>20} {a_mu:>11.2f}% {f1_mu:>11.2f}%")

stats_file.write_text('\n'.join(stats_lines) + '\n')
print(f"  -> {stats_file} (detailed stats)")
PYEOF

    python3 "$TMPPY" "$RUNS_DIR" "$BUILD_DIR/fh_results.txt"
    rm -f "$TMPPY"
fi

echo ""
echo "=============================================="
echo "  Fixed Hessian Benchmark Complete"
echo "=============================================="
echo "End time: $(date)"
echo "Console log: $BUILD_DIR/$LOG_FILE"
if [ "$NUM_RUNS" -gt 1 ]; then
    echo "Per-run results: $BUILD_DIR/runs/"
    echo "Stats detail:   $BUILD_DIR/runs/fh_results_stats.txt"
fi
echo "Result file:    $BUILD_DIR/fh_results.txt"
