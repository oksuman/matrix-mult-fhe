# matrix-mult-fhe

Secure matrix multiplication algorithms using the CKKS scheme.

## Overview

This project implements various secure matrix multiplication algorithms using the CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme through OpenFHE library. It includes benchmarking tools, tests, and a linear regression application example.

## Requirements

### Prerequisites
- C++ Compiler (gcc/g++ >= 9.4.0)
- CMake (>= 3.5.1)
- Make
- Git
- OpenFHE Library

### Installing OpenFHE
The OpenFHE library must be installed on your system. You can find:
- Source code: [OpenFHE GitHub Repository](https://github.com/openfheorg/openfhe-development)
- Installation guide: [OpenFHE Documentation](https://openfhe-development.readthedocs.io)

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/oksuman/matrix-mult-fhe.git
cd matrix-mult-fhe
```

If you already cloned the repository without `--recursive`, run:
```bash
git submodule init
git submodule update
```

2. Create and enter build directory:
```bash
mkdir build
cd build
```

3. Configure and build the project:
```bash
cmake ..
make
```

## Project Structure

```
matrix-mult-fhe/
├── CMakeLists.txt                    # Main CMake configuration file
├── app/
│   └── linear_regression/            # Linear regression application example
├── benchmark/                        # Benchmarking tools
│   ├── deep_multiplication/
│   ├── inversion/
│   └── single_multiplication/
├── external/                         # External dependencies
│   ├── benchmark/                    # Google Benchmark library
│   └── googletest/                   # Google Test framework
├── src/                             # Core library source files
│   ├── encryption.cpp
│   ├── encryption.h
│   ├── mat_inv.h
│   └── matrix_algo_*.h
├── tests/                           # Test files
│   ├── inverse_*_test.cpp
│   └── mult_*_test.cpp
└── utils/                           # Utility headers
    ├── csv_processor.h
    ├── diagonal_packing.h
    └── matrix_utils.h
```

## Running Tests

To build and run all tests:
```bash
cd build
make
ctest
```

To run specific test categories:
```bash
# Run multiplication tests
./tests/mult_*_test

# Run inversion tests
./tests/inverse_*_test
```

## Running Benchmarks

The project includes several benchmark suites:
```bash
# Single multiplication benchmarks
cd build/benchmark/single_multiplication
./run_benchmarks.sh

# Deep multiplication benchmarks
cd build/benchmark/deep_multiplication
./run_squaring_benchmarks.sh

# Matrix inversion benchmarks
cd build/benchmark/inversion
./run_inversion_benchmarks.sh
```

## Linear Regression Example

The project includes a linear regression application example that demonstrates the use of secure matrix operations:
```bash
cd build/app/linear_regression
./linear_regression
```
## References
This project implements matrix multiplication algorithms from the following papers:

### JKLS18
Xiaoqian Jiang, Miran Kim, Kristin E. Lauter, and Yongsoo Song. "Secure Outsourced Matrix Computation and Application to Neural Networks." In *Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS 2018)*, pages 1209-1222, 2018. [https://eprint.iacr.org/2018/1041.pdf]

### RT22
Panagiotis Rizomiliotis and Aikaterini Triakosia. "On Matrix Multiplication with Homomorphic Encryption." In *Proceedings of the 2022 on Cloud Computing Security Workshop (CCSW 2022)*, pages 53-61, 2022. [https://dl.acm.org/doi/10.1145/3560810.3564267]

### AS24
Aikata Aikata and Sujoy Sinha Roy. "Secure and Efficient Outsourced Matrix Multiplication with Homomorphic Encryption." To appear in the proceedings of INDOCRYPT 2024, 2024. [https://eprint.iacr.org/2024/1730.pdf]
