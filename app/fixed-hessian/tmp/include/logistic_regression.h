#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"
#include <iostream>
#include <vector>
#include <random>

class LogisticRegression {
private:
    Vector weights;
    
    // Helper functions (HE-friendly: only +, -, *)
    Vector mat_vec_mult(const Matrix& A, const Vector& x);
    Vector mat_T_vec_mult(const Matrix& A, const Vector& x);
    Vector solve_linear(Matrix A, Vector b);  // Only for plaintext verification
    
    // Mini-batch helper functions
    std::vector<int> get_random_batch_indices(int n_samples, int batch_size);
    void extract_batch(const Matrix& X_full, const Vector& y_full,
                      const std::vector<int>& indices,
                      Matrix& X_batch, Vector& y_batch);
    
    // BV18 functions (HE-compatible)
    double sigmoid_bv18(double z);  // σ(x) ≈ 1/2 + x/4
    Matrix compute_hessian_bv18(const Matrix& X, const Vector& y);
    Vector compute_diagonal_hessian_bv18(const Matrix& X, const Vector& y);
    Vector approximate_inverse_bv18(const Vector& diag_H, int n_samples, int n_features);
    
public:
    LogisticRegression(int n_features);
    
    // BV18 methods
    // All use FIXED iterations (no convergence check - not HE-friendly)
    // Note: tol parameter kept for compatibility but ignored
    
    void fit_hessian_bv18(const Matrix& X, const Vector& y,
                          int max_iter = 100, double tol = 0, int batch_size = -1);
    
    void fit_fixed_hessian_bv18(const Matrix& X, const Vector& y,
                                int max_iter = 100, double tol = 0, int batch_size = -1);
    
    // Simplified Fixed Hessian (SFH) - MOST HE-FRIENDLY
    void fit_fixed_hessian_diagonal_bv18(const Matrix& X, const Vector& y,
                                         int max_iter = 100, double tol = 0, int batch_size = -1);
    
    Vector predict_proba_bv18(const Matrix& X);
    Vector get_weights();
};

#endif