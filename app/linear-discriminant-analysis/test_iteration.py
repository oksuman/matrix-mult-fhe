#!/usr/bin/env python3
"""
Plaintext iteration analysis for scalar inverse and matrix inverse.
Uses Z-score normalization with α_upper = N * d
"""

import numpy as np
import pandas as pd

# Load data
train_df = pd.read_csv('data/heart_disease_train.csv')
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

N, d = X.shape
print(f"N (samples): {N}")
print(f"d (features): {d}")
print(f"α_upper = N * d = {N * d}")

# Z-score normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / (X_std + 1e-10)

print(f"\nZ-score normalized X: mean={X_norm.mean():.6f}, std={X_norm.std():.6f}")

# Compute S_W
def compute_Sw(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        Xc_centered = Xc - mu_c
        Sw += Xc_centered.T @ Xc_centered
    return Sw

Sw = compute_Sw(X_norm, y)
trace_Sw = np.trace(Sw)
alpha_upper = N * d

print(f"\ntrace(S_W) = {trace_Sw:.4f}")
print(f"α_upper = {alpha_upper}")
print(f"ratio = α_upper / trace = {alpha_upper / trace_Sw:.4f}")
print(f"a_bar_init = 1 - trace/α_upper = {1 - trace_Sw / alpha_upper:.6f}")

# Scalar inverse iteration analysis
print("\n" + "="*60)
print("Scalar Inverse Iteration Analysis")
print("="*60)

def scalar_inverse_analysis(a, alpha_upper, max_iter=20):
    """Analyze convergence of scalar inverse"""
    y = 1.0 / alpha_upper
    a_bar = 1.0 - a / alpha_upper
    expected = 1.0 / a

    print(f"{'Iter':<6} {'y':<18} {'rel_error':<12} {'a_bar':<12}")
    print("-" * 50)

    for k in range(max_iter):
        y = y * (1 + a_bar)
        a_bar = a_bar * a_bar

        rel_error = abs(y - expected) / expected
        print(f"{k:<6} {y:<18.12f} {rel_error:<12.2e} {a_bar:<12.2e}")

        if rel_error < 1e-12:
            print(f"\nConverged at iteration {k+1}")
            return k + 1

    return max_iter

iters_scalar = scalar_inverse_analysis(trace_Sw, alpha_upper)

# Matrix inverse iteration analysis
print("\n" + "="*60)
print("Matrix Inverse Iteration Analysis")
print("="*60)

def matrix_inverse_analysis(A, alpha, max_iter=20):
    """Analyze convergence of matrix inverse"""
    n = A.shape[0]
    I = np.eye(n)

    # Use 1/alpha directly (in encrypted version, this comes from scalar_inverse)
    t_inv = 1.0 / alpha

    Y = t_inv * I
    A_bar = I - t_inv * A

    A_inv_expected = np.linalg.inv(A)

    print(f"Initial ||A_bar||_max = {np.abs(A_bar).max():.6f}")
    print(f"\n{'Iter':<6} {'max_error':<14} {'||A_bar||_max':<14}")
    print("-" * 36)

    for k in range(max_iter):
        Y = Y @ (I + A_bar)
        A_bar = A_bar @ A_bar

        max_error = np.abs(Y - A_inv_expected).max()
        a_bar_max = np.abs(A_bar).max()

        print(f"{k:<6} {max_error:<14.2e} {a_bar_max:<14.2e}")

        if max_error < 1e-10:
            print(f"\nConverged at iteration {k+1}")
            return k + 1

    return max_iter

# Pad S_W to 16x16 (as in encrypted version)
d_padded = 16
Sw_padded = np.zeros((d_padded, d_padded))
Sw_padded[:d, :d] = Sw

# Add average diagonal value to padding to ensure convergence
# A_bar[i,i] = 1 - Sw[i,i]/trace -> need Sw[i,i] > 0 for A_bar < 1
avg_diag = trace_Sw / d  # Use original trace / original d
print(f"\nPadding diagonal with avg_diag = trace/d = {avg_diag:.4f}")
for i in range(d, d_padded):
    Sw_padded[i, i] = avg_diag

trace_padded = np.trace(Sw_padded)
print(f"New trace after padding = {trace_padded:.4f}")
print(f"\nPadded S_W ({d_padded}x{d_padded}): trace = {trace_padded:.4f}")

iters_matrix = matrix_inverse_analysis(Sw_padded, trace_padded)

# Combined analysis
print("\n" + "="*60)
print("Combined Analysis (Scalar + Matrix)")
print("="*60)

def combined_inverse(A, alpha_upper, scalar_iters, matrix_iters):
    """Full inverse computation with specified iterations"""
    n = A.shape[0]
    I = np.eye(n)
    trace_A = np.trace(A)

    # Step 1: Scalar inverse of trace
    y = 1.0 / alpha_upper
    a_bar = 1.0 - trace_A / alpha_upper
    for _ in range(scalar_iters):
        y = y * (1 + a_bar)
        a_bar = a_bar * a_bar
    t_inv = y

    # Step 2: Matrix inverse
    Y = t_inv * I
    A_bar = I - t_inv * A
    for _ in range(matrix_iters):
        Y = Y @ (I + A_bar)
        A_bar = A_bar @ A_bar

    return Y, t_inv

# Test with different iteration counts
print(f"\nTesting with α_upper = N * d = {alpha_upper}")
print(f"(Note: padded trace = {trace_padded:.4f})")

A_inv_expected = np.linalg.inv(Sw_padded)

for s_iter in [4, 5, 6, 7, 8]:
    for m_iter in [8, 10, 12, 14]:
        Y, t_inv = combined_inverse(Sw_padded, alpha_upper, s_iter, m_iter)
        max_error = np.abs(Y - A_inv_expected).max()

        # Also check S_W @ S_W^{-1} = I
        product_error = np.abs(Sw_padded @ Y - np.eye(d_padded)).max()

        if max_error < 1e-6:
            print(f"scalar_iter={s_iter}, matrix_iter={m_iter}: max_error={max_error:.2e}, ||SW @ SW^-1 - I||={product_error:.2e} ✓")

# Recommended iterations
print("\n" + "="*60)
print("Recommendation")
print("="*60)
print(f"For Z-score normalization with α_upper = N*d = {alpha_upper}:")
print(f"  - Scalar inverse iterations: {iters_scalar}")
print(f"  - Matrix inverse iterations: {iters_matrix}")
print(f"  - Total depth per scalar iter: 2 (mult + mult)")
print(f"  - Total depth per matrix iter: 1 (matrix mult is parallel)")
