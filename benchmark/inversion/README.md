# Matrix Inversion Benchmark

Measures time and accuracy of encrypted matrix inversion across 5 algorithms and dimensions d = {4, 8, 16, 32, 64}.

## Algorithms

- Naive: d = {4, 8} only (d^2 ciphertexts, no bootstrapping)
- NewCol, AR24, JKLS18: d = {4, 8, 16, 32, 64}
- RT22: d = {4, 8, 16, 32} only (cubic batch size prevents d=64)

## Two Variants

### Original (trace + eval_scalar_inverse)
- `MULT_DEPTH=31`, `SCALE_MOD_SIZE=59`, `FIRST_MOD_SIZE=60`
- Bootstrapping: `levelBudget={4,5}`, `numIterations=2`, `precision=18`
- Scalar inverse iterations: 1

### Simple (1/d² upperbound scaling, no trace)
- `MULT_DEPTH=30`, `SCALE_MOD_SIZE=59`, `FIRST_MOD_SIZE=60`
- Bootstrapping: `levelBudget={4,4}`, `numIterations=2`, `precision=18`

## Inversion Iterations

| d   | 4  | 8  | 16 | 32 | 64 |
|-----|----|----|----|----|-----|
| r   | 18 | 22 | 25 | 28 | 31 |

Seed: `1000 + run`

## How to Run

```bash
cd build/benchmark/inversion

# Simple variant only (default)
./run_inversion_benchmarks.sh [num_runs]

# Original variant only
./run_inversion_benchmarks.sh [num_runs] original

# Both variants
./run_inversion_benchmarks.sh [num_runs] all
```

## Output

- `inversion_console.log` — raw console output
- `inversion_results.txt` — clean results: parameters, time table, accuracy table

`inversion_results.txt` is generated automatically via `generate_summary_table.sh`.
