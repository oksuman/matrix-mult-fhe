# Matrix Operation Benchmarks

This directory contains benchmarking code for various matrix operations implemented with OpenFHE. The benchmarks are organized into different categories:

## Available Benchmarks

### 1. Single Matrix Multiplication (`single_multiplication/`)
Benchmarks for single matrix multiplication operations, comparing different algorithms:
- JKLS18 (CCS'18)
- RT22 (CCSW'22)
  - Regular multiplication for matrices up to 16x16
  - Strassen algorithm for 32x32 matrices using 16x16 blocks
- AS24 (Indocrypt'24)
- NewCol
- NewRow
- DP (Diagonal Packing Method)
  
### 2. Deep Matrix Multiplication (`deep_multiplication/`)
Benchmarks for repeated matrix squaring operations (A -> A^2 -> A^4 -> A^8 -> ...), comparing different algorithms:
- JKLS18 (CCS'18)
- RT22 (CCSW'22)
  - Regular multiplication for matrices up to 32x32
  - Strassen algorithm for 64x64 matrices using 32x32 blocks
- AS24 (Indocrypt'24)
- NewCol
- NewRow
- DP (Diagonal Packing Method)

Each algorithm performs 10 rounds of squaring operations with increased multiplicative depth.

### 3. Matrix Inversion (`inversion/`) 
Benchmarks for matrix inversion algorithms.
- Naive Approach
- JKLS18 (CCS'18)
- RT22 (CCSW'22)
  - Regular multiplication for matrices up to 32x32
  - Strassen algorithm for 64x64 matrices using 32x32 blocks
- AS24 (Indocrypt'24)
- NewCol
- NewRow
- DP (Diagonal Packing Method)
  
## Build Instructions

1. Create and navigate to build directory:
```bash
mkdir build
cd build
```

2. Configure CMake:
```bash
cmake ..
```

3. Build the benchmarks:
```bash
make
```

This will build all benchmark executables in their respective directories:
- `build/benchmark/single_multiplication/`
- `build/benchmark/deep_multiplication/`
- `build/benchmark/inversion/`

## Running Benchmarks

### Single Matrix Multiplication

1. Navigate to the benchmark directory:
```bash
cd build/benchmark/single_multiplication
```

2. Grant execution permission to the script:
```bash
chmod +x run_benchmarks.sh
```

3. Run the benchmarks:
```bash
./run_benchmarks.sh
```

The script will:
- Run each algorithm's benchmark separately
- Provide system stabilization time between runs
- Generate result files for each algorithm

### Deep Matrix Multiplication

1. Navigate to the benchmark directory:
```bash
cd build/benchmark/deep_multiplication
```

2. Grant execution permission to the script:
```bash
chmod +x run_squaring_benchmarks.sh
```

3. Run the benchmarks:
```bash
./run_squaring_benchmarks.sh
```

### Inversion 

1. Navigate to the benchmark directory:
```bash
cd build/benchmark/inversion
```

2. Grant execution permission to the script:
```bash
chmod +x run_inversion_benchmarks.sh
```

3. Run the benchmarks:
```bash
./run_inversion_benchmarks.sh
```

The script will:
- Run each algorithm's squaring benchmark separately
- Allow system cooling between runs
- Generate individual result files for each algorithm

## System Requirements

- CMake 3.5.1 or higher
- C++17 compiler
- OpenFHE library
- Google Benchmark library (automatically fetched by CMake)

## Adding New Benchmarks

To add new benchmarks:
1. Create a new directory under `benchmark/`
2. Add appropriate CMakeLists.txt
3. Implement benchmark code following existing patterns
4. Update main CMakeLists.txt to include new directory
5. Update this README with new benchmark details
