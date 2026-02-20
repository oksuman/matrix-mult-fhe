# matrix-mult-fhe

Secure matrix multiplication and inversion algorithms using the CKKS homomorphic encryption scheme via OpenFHE.
The project includes matrix inversion via iterative refinement, three ML application experiments (Linear Regression, LDA, Fixed Hessian), and comprehensive benchmarking suites.

## Requirements

- C++ Compiler (gcc/g++ >= 9.4.0)
- CMake (>= 3.5.1)
- OpenFHE Library ([GitHub](https://github.com/openfheorg/openfhe-development) | [Docs](https://openfhe-development.readthedocs.io))

## Installation

```bash
git clone --recursive https://github.com/oksuman/matrix-mult-fhe.git
cd matrix-mult-fhe

# If cloned without --recursive:
git submodule init && git submodule update

# Build
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

## Project Structure

```
matrix-mult-fhe/
├── src/                             # Core library
│   ├── encryption.{h,cpp}           # CKKS encryption wrapper
│   ├── matrix_algo_singlePack.h     # JKLS18, RT22, AR24, NewCol, NewRow
│   ├── matrix_inversion_algo.h      # Iterative matrix inversion
│   ├── matrix_algo_multiPack.h      # Diagonal-packed multiplication
│   ├── naive_inversion.h            # Element-wise baseline
│   └── rotation.h                   # Power-of-2 rotation key management
├── tests/              
├── benchmark/
│   ├── squaring/                    # Deep (iterated) squaring benchmarks
│   ├── inversion/                   # Matrix inversion benchmarks
│   ├── app/                         # ML application benchmark scripts
│   ├── benchmark_config.h           # Unified benchmark parameters
│   └── run_all_benchmarks.sh        # Master benchmark runner
├── app/
│   ├── linear-regression/           # Encrypted linear regression 
│   ├── linear-discriminant-analysis/ # Encrypted LDA 
│   ├── fixed-hessian/               # Encrypted fixed Hessian for logisitic regresion
│   └── common/                      # Shared evaluation metrics
├── utils/                           # Matrix utilities, CSV processing
└── external/                        # Google Test & Benchmark submodules
```

## Running Tests

```bash
cd build
ctest                              # Run all tests
./tests/mult_ar24_test             # Run a specific test
```

## Running Benchmarks

```bash
# Deep multiplication & inversion (scripts are copied to build dir by CMake):
cd build/benchmark/squaring && ./run_squaring_benchmarks.sh [num_runs]
cd build/benchmark/inversion && ./run_inversion_benchmarks.sh [num_runs]

# Application benchmarks (run from source dir, executables are found via relative paths):
cd benchmark/app && ./run_lr.sh && ./run_lda.sh && ./run_fh.sh

# Or run everything at once (from source dir):
cd benchmark && ./run_all_benchmarks.sh
```

See `benchmark/*/README.md` for details on each experiment.

## References
- **JKLS18**: X. Jiang, M. Kim, K. Lauter, Y. Song. "Secure Outsourced Matrix Computation and Application to Neural Networks." CCS 2018. [ePrint](https://eprint.iacr.org/2018/1041.pdf)
- **RT22**: P. Rizomiliotis, A. Triakosia. "On Matrix Multiplication with Homomorphic Encryption." CCSW 2022. [ACM](https://dl.acm.org/doi/10.1145/3560810.3564267)
- **AR24**: A. Aikata, S.S. Roy. "Secure and Efficient Outsourced Matrix Multiplication with Homomorphic Encryption." INDOCRYPT 2024. [ePrint](https://eprint.iacr.org/2024/1730.pdf)
