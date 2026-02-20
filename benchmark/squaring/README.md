# Deep Matrix Multiplication Benchmark

Measures time and accuracy of iterated matrix squaring (M^{2^k}) across 5 algorithms and dimensions d = {4, 8, 16, 32, 64}.

## Algorithms

- Naive, NewCol, AR24, JKLS18: d = {4, 8, 16, 32, 64}
- RT22: d = {4, 8, 16, 32} only (cubic batch size prevents d=64)

## Parameters

- `SCALE_MOD_SIZE=50`, `FIRST_MOD_SIZE=60`
- `SQUARING_ITERATIONS=15`
- Bootstrapping: `levelBudget={4,4}`, `numIterations=2`, `precision=18`

## How to Run

```bash
cd build/benchmark/squaring
./run_squaring_benchmarks.sh [num_runs]    # default: 1 run per dimension
```

## Output

- `squaring_benchmark_results.txt` — raw output
- `squaring_summary_time.txt` — time comparison table
- `squaring_summary_memory.txt` — memory usage table
- `squaring_summary_accuracy.txt` — accuracy (log2 error) table

Summary tables are generated automatically via `generate_summary_table.sh`.
