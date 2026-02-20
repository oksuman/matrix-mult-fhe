# Application Benchmarks

Three ML applications comparing encrypted matrix operations.

## Experiments

### Linear Regression (`run_lr.sh`)
- Dataset: Boston Housing (64 train, 256 test, 8 features)
- Compares: Naive, NewCol, AR24
- Matrix inversion: 8x8, r=18, multDepth=30

### LDA (`run_lda.sh`)
- Dataset: Heart Disease (64 train, 128 test, 13 features padded to 16)
- Compares: NewCol, AR24
- Matrix inversion: 16x16, r=25, multDepth=30

### Fixed Hessian (`run_fh.sh`)
- Dataset: Diabetes (64 train, 256 test)
- Two modes:
  - **FH** (Full Hessian): uses AR24 or NewCol for 16x16 matrix inversion (r=22)
  - **SFH** (Simplified): diagonal approximation, no full matrix inversion, no AR24/NewCol comparison
- multDepth=30

## How to Run

```bash
# Run from source directory (scripts find executables via relative paths):
cd benchmark/app

# Run all three:
./run_lr.sh && ./run_lda.sh && ./run_fh.sh

# Or run individually:
./run_lr.sh    # Linear Regression
./run_lda.sh   # LDA
./run_fh.sh    # Fixed Hessian
```

All scripts enforce `OMP_NUM_THREADS=1` and include 30-second cooling periods between runs.
