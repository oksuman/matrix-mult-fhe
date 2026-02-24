# Application Benchmarks

Three ML applications comparing encrypted matrix inversion algorithms (NewCol vs AR24).

## Experiments

### Linear Regression (`run_lr.sh`)
- Dataset: California Housing (64 train, 8 features)
- Compares: Naive, AR24, NewCol
- Matrix inversion: 8×8, r=18, multDepth=31, bootstrapping enabled

### LDA (`run_lda.sh`)
- Dataset: Heart Disease Cleveland (64 train, 13 features padded to 16)
- Compares: NewCol, AR24
- Matrix inversion: 16×16, r=25, multDepth=31, bootstrapping enabled

### Fixed Hessian (`run_fh.sh`)
- Dataset: Diabetes (64 train, 256 test, 8 features padded to 16)
- Three modes run in a single executable:
  - **FH (AR24)**: full 16×16 matrix inversion with AR24, r=22
  - **FH (NewCol)**: full 16×16 matrix inversion with NewCol, r=22
  - **SFH**: diagonal Hessian approximation (no full matrix inversion)
- multDepth=31, bootstrapping enabled

## How to Run

### Run all three apps sequentially (recommended)

```bash
# From project root — does clean build then runs LDA → LR → FH
bash benchmark/app/run_all_apps.sh
```

빌드가 되어 있어야 함. 처음 실행 시:
```bash
mkdir build && cd build && cmake .. && make -j$(sysctl -n hw.physicalcpu) && cd ..
bash benchmark/app/run_all_apps.sh
```

### Run individually

```bash
# From project root
bash benchmark/app/run_lr.sh
bash benchmark/app/run_lda.sh
bash benchmark/app/run_fh.sh
```

## Output Files

Each app produces two files in its build directory:

| App | Result file | Console log |
|-----|------------|-------------|
| LDA | `build/app/linear-discriminant-analysis/lda_results.txt` | `lda_console.log` |
| LR  | `build/app/linear-regression/lr_results.txt` | `lr_console.log` |
| FH  | `build/app/fixed-hessian/fh_results.txt` | `fh_console.log` |

- **`*_results.txt`**: clean comparison table (timing + accuracy)
- **`*_console.log`**: full console output including intermediate steps

## Notes

- `OMP_NUM_THREADS` defaults to physical CPU core count; override with `OMP_NUM_THREADS=8 bash run_lr.sh`
- 30-second cooling periods between runs within each script
- 60-second cooling between apps in `run_all_apps.sh`
- Bootstrap threshold differences between NewCol and AR24 are **intentional** (not a fairness issue)
