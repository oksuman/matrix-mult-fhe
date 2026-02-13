#include "logistic_regression.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

LogisticRegression::LogisticRegression(int n_features) : weights(n_features) {
    for (int i = 0; i < n_features; i++) {
        weights[i] = 0.001;  // BV18 initialization
    }
}

// BV18: Sigmoid approximation σ(z) ≈ 1/2 + z/4
double LogisticRegression::sigmoid_bv18(double z) {
    return 0.5 + z / 4.0;
}

Vector LogisticRegression::mat_vec_mult(const Matrix& A, const Vector& x) {
    Vector result(A.rows);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            result[i] += A(i, j) * x[j];
        }
    }
    return result;
}

Vector LogisticRegression::mat_T_vec_mult(const Matrix& A, const Vector& x) {
    Vector result(A.cols);
    for (int j = 0; j < A.cols; j++) {
        for (int i = 0; i < A.rows; i++) {
            result[j] += A(i, j) * x[i];
        }
    }
    return result;
}

Vector LogisticRegression::solve_linear(Matrix A, Vector b) {
    int n = A.rows;
    
    for (int k = 0; k < n; k++) {
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            if (std::abs(A(i, k)) > std::abs(A(max_row, k))) {
                max_row = i;
            }
        }
        std::swap(A.data[k], A.data[max_row]);
        std::swap(b[k], b[max_row]);
        
        for (int i = k + 1; i < n; i++) {
            if (std::abs(A(k, k)) < 1e-10) continue;
            double factor = A(i, k) / A(k, k);
            for (int j = k; j < n; j++) {
                A(i, j) -= factor * A(k, j);
            }
            b[i] -= factor * b[k];
        }
    }
    
    Vector x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A(i, j) * x[j];
        }
        if (std::abs(A(i, i)) > 1e-10) {
            x[i] /= A(i, i);
        }
    }
    return x;
}

// Mini-batch: Generate random batch indices
std::vector<int> LogisticRegression::get_random_batch_indices(int n_samples, int batch_size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (int i = n_samples - 1; i > 0; i--) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        std::swap(indices[i], indices[j]);
    }
    
    // Select first batch_size elements
    indices.resize(batch_size);
    return indices;
}

// Mini-batch: Extract batch from full dataset
// x_full: n×f, y_full: n×1 → X_batch: b×f, y_batch: b×1
void LogisticRegression::extract_batch(const Matrix& x_full, const Vector& y_full,
                                       const std::vector<int>& indices,
                                       Matrix& X_batch, Vector& y_batch) {
    int b = indices.size();
    int f = x_full.cols;
    
    for (int i = 0; i < b; i++) {
        int idx = indices[i];
        for (int j = 0; j < f; j++) {
            X_batch(i, j) = x_full(idx, j);
        }
        y_batch[i] = y_full[idx];
    }
}

// BV18: Full Hessian H = -1/4 * X^T * X
// Input: X (b×f matrix), y (b×1 vector)
// Output: H (f×f matrix)
Matrix LogisticRegression::compute_hessian_bv18(const Matrix& X, const Vector& y) {
    int b = X.rows, f = X.cols;  // b: batch_size, f: features
    Matrix H(f, f);
    
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            double sum = 0.0;
            for (int k = 0; k < b; k++) {
                sum += X(k, i) * X(k, j);
            }
            H(i, j) = -0.25 * sum;
        }
    }
    
    return H;
}

// BV18: Diagonal Hessian diag(H)_jj = -1/4 * Σᵢ(X_ij * Σₖ X_ik)
// Input: X (b×f matrix), y (b×1 vector)
// Output: diag_H (f×1 vector)
Vector LogisticRegression::compute_diagonal_hessian_bv18(const Matrix& X, const Vector& y) {
    int b = X.rows, f = X.cols;  // b: batch_size, f: features
    Vector diag_H(f);
    
    // Compute Σₖ X_ik for each sample i in batch
    std::vector<double> sum_vec(b);
    for (int i = 0; i < b; i++) {
        double sum = 0.0;
        for (int j = 0; j < f; j++) {
            sum += X(i, j);
        }
        sum_vec[i] = sum;
    }
    
    // Compute diag(H)_jj = -1/4 * Σᵢ(X_ij * sum_vec[i])
    for (int j = 0; j < f; j++) {
        double temp = 0.0;
        for (int i = 0; i < b; i++) {
            temp += X(i, j) * sum_vec[i];
        }
        diag_H[j] = -0.25 * temp;
    }
    
    return diag_H;
}

// BV18: Inverse approximation
// Input: diag_H (f×1 vector)
// Output: inv_diag_H (f×1 vector)
Vector LogisticRegression::approximate_inverse_bv18(const Vector& diag_H, int n_samples, int n_features) {
    int f = diag_H.size;
    Vector inv_diag_H(f);
    
    // Plaintext: exact inverse with regularization
    for (int j = 0; j < f; j++) {
        double h_val = diag_H[j];
        
        // Add small regularization for numerical stability
        if (std::abs(h_val) < 1e-10) {
            inv_diag_H[j] = 0.0;
        } else {
            inv_diag_H[j] = 1.0 / h_val;
        }
    }
    
    return inv_diag_H;
}

// BV18: Full Hessian method (recompute H each iteration)
// Input: x (n×f full dataset), y (n×1 labels)
// Uses mini-batch: random b samples per iteration
void LogisticRegression::fit_hessian_bv18(const Matrix& x, const Vector& y,
                                          int max_iter, double tol, int batch_size) {
    int n = x.rows, f = x.cols;  // n: total samples, f: features
    
    bool use_batch = (batch_size > 0 && batch_size < n);
    int b = use_batch ? batch_size : n;  // b: effective batch size
    
    std::cout << "=== BV18: Full Hessian ===" << std::endl;
    std::cout << "Iterations: " << max_iter << " (fixed)" << std::endl;
    std::cout << "Sigmoid: σ(x) ≈ 1/2 + x/4" << std::endl;
    
    if (use_batch) {
        std::cout << "Mini-batch mode: b=" << b << std::endl;
    } else {
        std::cout << "Full-batch mode: b=" << n << std::endl;
    }
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Get current batch
        Matrix X(b, f);
        Vector y_batch(b);
        
        if (use_batch) {
            std::vector<int> indices = get_random_batch_indices(n, b);
            extract_batch(x, y, indices, X, y_batch);
        } else {
            X = x;
            y_batch = y;
        }
        
        // z = X * weights (b×1)
        Vector z = mat_vec_mult(X, weights);
        
        // BV18 Gradient: g = Σᵢ[(1/2 - 1/4 * yᵢ * β^T * xᵢ) * yᵢ * xᵢ]
        Vector gradient(f);
        for (int j = 0; j < f; j++) {
            double sum = 0.0;
            for (int i = 0; i < b; i++) {
                double coef = (0.5 - 0.25 * y_batch[i] * z[i]) * y_batch[i];
                sum += coef * X(i, j);
            }
            gradient[j] = sum;
        }
        
        // Compute Hessian H (f×f)
        Matrix H = compute_hessian_bv18(X, y_batch);
        
        // Add regularization
        for (int i = 0; i < f; i++) {
            H(i, i) += 1e-5;
        }
        
        // Solve H * delta = gradient
        Vector delta = solve_linear(H, gradient);
        
        // Update: β = β - H^{-1} * g
        for (int i = 0; i < f; i++) {
            weights[i] -= delta[i];
        }
        
        if ((iter + 1) % 10 == 0 || iter == 0) {
            std::cout << "Iteration " << (iter + 1) << " completed" << std::endl;
        }
    }
    
    std::cout << "Training completed after " << max_iter << " iterations" << std::endl;
}

// BV18: Fixed Hessian method (compute H once, reuse)
// Input: x (n×f full dataset), y (n×1 labels)
// H computed on full dataset, gradient uses mini-batch
void LogisticRegression::fit_fixed_hessian_bv18(const Matrix& x, const Vector& y,
                                                int max_iter, double tol, int batch_size) {
    int n = x.rows, f = x.cols;  // n: total samples, f: features
    
    bool use_batch = (batch_size > 0 && batch_size < n);
    int b = use_batch ? batch_size : n;  // b: effective batch size
    
    std::cout << "=== BV18: Fixed Hessian ===" << std::endl;
    std::cout << "Iterations: " << max_iter << " (fixed)" << std::endl;
    std::cout << "Sigmoid: σ(x) ≈ 1/2 + x/4" << std::endl;
    
    if (use_batch) {
        std::cout << "Mini-batch gradient: b=" << b << std::endl;
        std::cout << "Fixed Hessian: computed on full dataset (n=" << n << ")" << std::endl;
    } else {
        std::cout << "Full-batch mode: b=" << n << std::endl;
    }
    
    // Compute Fixed Hessian H once on FULL dataset
    Matrix H = compute_hessian_bv18(x, y);
    
    for (int i = 0; i < f; i++) {
        H(i, i) += 1e-5;
    }
    
    std::cout << "Fixed Hessian computed" << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Get current batch for gradient
        Matrix X(b, f);
        Vector y_batch(b);
        
        if (use_batch) {
            std::vector<int> indices = get_random_batch_indices(n, b);
            extract_batch(x, y, indices, X, y_batch);
        } else {
            X = x;
            y_batch = y;
        }
        
        // z = X * weights (b×1)
        Vector z = mat_vec_mult(X, weights);
        
        // BV18 Gradient on current batch
        Vector gradient(f);
        for (int j = 0; j < f; j++) {
            double sum = 0.0;
            for (int i = 0; i < b; i++) {
                double coef = (0.5 - 0.25 * y_batch[i] * z[i]) * y_batch[i];
                sum += coef * X(i, j);
            }
            gradient[j] = sum;
        }
        
        // Solve H * delta = gradient (H is fixed)
        Vector delta = solve_linear(H, gradient);
        
        // Update: β = β - H^{-1} * g
        for (int i = 0; i < f; i++) {
            weights[i] -= delta[i];
        }
        
        if ((iter + 1) % 10 == 0 || iter == 0) {
            std::cout << "Iteration " << (iter + 1) << " completed" << std::endl;
        }
    }
    
    std::cout << "Training completed after " << max_iter << " iterations" << std::endl;
}

// BV18: Simplified Fixed Hessian (SFH) - MOST HE-FRIENDLY
// Input: x (n×f full dataset), y (n×1 labels)
// diag(H) computed on full dataset, gradient uses mini-batch
void LogisticRegression::fit_fixed_hessian_diagonal_bv18(const Matrix& x, const Vector& y,
                                                         int max_iter, double tol, int batch_size) {
    int n = x.rows, f = x.cols;  // n: total samples, f: features
    
    bool use_batch = (batch_size > 0 && batch_size < n);
    int b = use_batch ? batch_size : n;  // b: effective batch size
    
    std::cout << "=== BV18: Simplified Fixed Hessian (SFH) ===" << std::endl;
    std::cout << "Iterations: " << max_iter << " (fixed)" << std::endl;
    std::cout << "Sigmoid: σ(x) ≈ 1/2 + x/4" << std::endl;
    std::cout << "Most HE-friendly method" << std::endl;
    
    if (use_batch) {
        std::cout << "Mini-batch gradient: b=" << b << std::endl;
        std::cout << "Fixed Diagonal Hessian: computed on full dataset (n=" << n << ")" << std::endl;
    } else {
        std::cout << "Full-batch mode: b=" << n << std::endl;
    }
    
    // Compute Fixed Diagonal Hessian once on FULL dataset
    Vector diag_H = compute_diagonal_hessian_bv18(x, y);
    
    // Approximate inverse
    Vector inv_diag_H = approximate_inverse_bv18(diag_H, n, f);
    
    std::cout << "Fixed Diagonal Hessian computed" << std::endl;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Get current batch for gradient
        Matrix X(b, f);
        Vector y_batch(b);
        
        if (use_batch) {
            std::vector<int> indices = get_random_batch_indices(n, b);
            extract_batch(x, y, indices, X, y_batch);
        } else {
            X = x;
            y_batch = y;
        }
        
        // z = X * weights (b×1)
        Vector z = mat_vec_mult(X, weights);
        
        // BV18 Gradient on current batch
        Vector gradient(f);
        for (int j = 0; j < f; j++) {
            double sum = 0.0;
            for (int i = 0; i < b; i++) {
                double y_beta_x = y_batch[i] * z[i];
                double sigmoid_approx = 0.5 - 0.25 * y_beta_x;
                double coef = sigmoid_approx * y_batch[i];
                sum += coef * X(i, j);
            }
            gradient[j] = sum;
        }
        
        // Update: β = β - diag(H)^{-1} * g (element-wise)
        Vector delta(f);
        for (int j = 0; j < f; j++) {
            delta[j] = inv_diag_H[j] * gradient[j];
        }
        
        for (int j = 0; j < f; j++) {
            weights[j] -= delta[j];
        }
        
        if ((iter + 1) % 10 == 0 || iter == 0) {
            std::cout << "Iteration " << (iter + 1) << " completed" << std::endl;
        }
    }
    
    std::cout << "Training completed after " << max_iter << " iterations" << std::endl;
}

Vector LogisticRegression::predict_proba_bv18(const Matrix& X) {
    Vector z = mat_vec_mult(X, weights);
    Vector probs(z.size);
    for (int i = 0; i < z.size; i++) {
        probs[i] = sigmoid_bv18(z[i]);
    }
    return probs;
}

Vector LogisticRegression::get_weights() { 
    return weights; 
}