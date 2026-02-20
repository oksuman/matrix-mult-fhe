# Matrix Inversion Benchmark

Measures time and accuracy of encrypted matrix inversion across 5 algorithms and dimensions d = {4, 8, 16, 32, 64}.

## Algorithms

- Naive: d = {4, 8} only (d^2 ciphertexts, no bootstrapping)
- NewCol, AR24, JKLS18: d = {4, 8, 16, 32, 64}
- RT22: d = {4, 8, 16, 32} only (cubic batch size prevents d=64)

## Parameters

- `MULT_DEPTH=36`, `SCALE_MOD_SIZE=59`, `FIRST_MOD_SIZE=60`
- Bootstrapping: `levelBudget={4,4}`, `numIterations=2`, `precision=18`
- Scalar inverse iterations: 1
- Inversion iterations: d=4:18, d=8:22, d=16:25, d=32:27, d=64:30
- Seed: `1000 + run`

## How to Run

```bash
cd build/benchmark/inversion
./run_inversion_benchmarks.sh [num_runs]    # default: 1 run per dimension
```

## Output

- `inversion_benchmark_results.txt` — raw output
- `inversion_summary_time.txt` — time comparison table
- `inversion_summary_accuracy.txt` — accuracy (log2 error) table

Summary tables are generated automatically via `generate_summary_table.sh`.
