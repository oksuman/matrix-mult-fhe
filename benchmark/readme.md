# Matrix Operation Benchmarks

Benchmarks for matrix operations with CKKS homomorphic encryption using OpenFHE.

## Quick Start

```bash
# Build
cd /path/to/matrix-mult-fhe
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 1. Deep Multiplication (`deep_multiplication/`)

Repeated squaring: A → A² → A⁴ → ... → A^(2^15)

### Algorithms

| Algorithm | Ciphertexts | Mult Depth | Description |
|-----------|-------------|------------|-------------|
| NewCol    | 1           | 2 per iter | Column-based packing |
| AR24      | 1           | 3 per iter | INDOCRYPT 2024 |
| JKLS18    | 1           | 3 per iter | CCS 2018 |
| RT22      | 1           | 2 per iter | CCSW 2022 |
| NewRow    | 1           | 2 per iter | Row-based packing |
| Diagonal  | d           | 1 per iter | d ciphertexts for d diagonals |

### Run

```bash
cd build/benchmark/deep_multiplication

# Run all algorithms (with 45s cooling between each)
./run_squaring_benchmarks.sh        # 1 trial per dimension
./run_squaring_benchmarks.sh 10     # 10 trials per dimension

# Run single algorithm
./benchmark_squaring_newcol [num_runs]
./benchmark_squaring_ar24 [num_runs]
./benchmark_squaring_jkls18 [num_runs]
./benchmark_squaring_rt22 [num_runs]
./benchmark_squaring_newrow [num_runs]
./benchmark_squaring_diag [num_runs]
```

Results saved to `squaring_benchmark_results.txt`.

## 2. Matrix Inversion (`inversion/`)

Newton-Schulz iteration: Y_{k+1} = Y_k(2I - AY_k)

### Iterations by Dimension

| Dimension | Iterations (r) |
|-----------|----------------|
| 4×4       | 18             |
| 8×8       | 21             |
| 16×16     | 25             |
| 32×32     | 28             |
| 64×64     | 31             |

### Run

```bash
cd build/benchmark/inversion

# Run all algorithms
./run_inversion_benchmarks.sh       # 1 trial
./run_inversion_benchmarks.sh 10    # 10 trials

# Run single algorithm
./benchmark_inversion_newcol [num_runs]
./benchmark_inversion_ar24 [num_runs]
./benchmark_inversion_jkls18 [num_runs]
./benchmark_inversion_rt22 [num_runs]
./benchmark_inversion_newrow [num_runs]
./benchmark_inversion_diag [num_runs]
./benchmark_inversion_naive [num_runs]  # d≤8 only
```

## Output Format

Each benchmark reports:

```
========== NewCol Squaring d=8 ==========
Mult Depth: 30, Scaling: 50, Ring: 131072
  [1/1] 30.29s, log2(err)=-38.0

--- Summary (d=8) ---
Time: 30.29s
  Frobenius Norm Error: 0.000000
  Relative Frobenius:   0.000000
  log2(Rel. Frob.):     -38.0

=== Memory Analysis ===
  Idle Memory:        0.0312 GB
  Setup Memory:       1.2456 GB
  Peak Memory:        1.8234 GB
  Setup Overhead:     1.2144 GB
  Compute Overhead:   0.5778 GB

=== Serialized Sizes ===
  Ciphertext:         125.50 MB
  Rotation Keys:      1024.00 MB
  Relin Key:          64.00 MB
```

## Fairness Guarantees

- **Single-thread**: `OMP_NUM_THREADS=1`
- **Cooling period**: 45 seconds between algorithms
- **Time measurement**: Computation only (excludes encryption/decryption/output)
- **Ground truth**: Plaintext computation for accuracy comparison

## Configuration

Common parameters in `benchmark_config.h`:

| Parameter | Value |
|-----------|-------|
| Security Level | HEStd_128_classic |
| Scaling Mod Size | 50 (deep mult), 59 (inversion) |
| Squaring Iterations | 15 |

## File Structure

```
benchmark/
├── README.md
├── benchmark_config.h      # Common configuration & ErrorMetrics
├── memory_tracker.h/cpp    # Memory monitoring utilities
│
├── deep_multiplication/
│   ├── benchmark_squaring.h
│   ├── benchmark_squaring_*.cpp
│   └── run_squaring_benchmarks.sh
│
└── inversion/
    ├── benchmark_inversion_*.cpp
    └── run_inversion_benchmarks.sh
```
