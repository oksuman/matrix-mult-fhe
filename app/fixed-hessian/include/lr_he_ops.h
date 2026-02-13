#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>

// CKKS-friendly plaintext operations for logistic regression
// Simulates SIMD operations that will be performed on encrypted data
// Based on LDA's lda_plaintext_ops.h, extended for LR-specific operations

class LROps {
public:
    // Rotate vector left by rotateIndex (positive = left, negative = right)
    static std::vector<double> rotate(const std::vector<double>& vec, int rotateIndex) {
        if (vec.empty()) return vec;
        std::vector<double> result = vec;
        int n = result.size();
        rotateIndex = ((rotateIndex % n) + n) % n;
        if (rotateIndex > 0) {
            std::rotate(result.begin(), result.begin() + rotateIndex, result.end());
        }
        return result;
    }

    // Element-wise addition
    static std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] + (i < b.size() ? b[i] : 0.0);
        }
        return result;
    }

    static void addInPlace(std::vector<double>& a, const std::vector<double>& b) {
        for (size_t i = 0; i < a.size() && i < b.size(); i++) {
            a[i] += b[i];
        }
    }

    // Element-wise subtraction
    static std::vector<double> sub(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] - (i < b.size() ? b[i] : 0.0);
        }
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    static std::vector<double> mult(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] * (i < b.size() ? b[i] : 0.0);
        }
        return result;
    }

    // Multiply by scalar
    static std::vector<double> multScalar(const std::vector<double>& a, double scalar) {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] * scalar;
        }
        return result;
    }

    // Apply mask
    static std::vector<double> mask(const std::vector<double>& vec, const std::vector<double>& maskVec) {
        return mult(vec, maskVec);
    }

    // Row folding sum: sum rows of a d×d matrix (packed row-major, length d*d)
    // After folding, row 0 contains the sum of all rows, replicated across all rows
    // Rotates by d, 2d, 4d, ..., d*d/2
    static std::vector<double> rowFoldingSum(const std::vector<double>& vec, int d) {
        std::vector<double> result = vec;
        for (int i = 1; i < d; i *= 2) {
            auto rotated = rotate(result, i * d);
            addInPlace(result, rotated);
        }
        return result;
    }

    // Column folding sum: sum within each row (across columns)
    // After folding, column 0 of each row contains the row sum
    // Rotates by 1, 2, 4, ..., d/2
    static std::vector<double> columnFoldingSum(const std::vector<double>& vec, int d) {
        std::vector<double> result = vec;
        for (int i = 1; i < d; i *= 2) {
            auto rotated = rotate(result, i);
            addInPlace(result, rotated);
        }
        return result;
    }

    // Mask column 0 only: keep position (i*d + 0) for each row i
    static std::vector<double> maskColumn0(const std::vector<double>& vec, int d) {
        int n = vec.size();
        std::vector<double> maskVec(n, 0.0);
        for (int i = 0; i < n; i += d) {
            maskVec[i] = 1.0;
        }
        return mult(vec, maskVec);
    }

    // Replicate column 0 to all columns (rotate by 1,2,4,...,d/2 and add)
    // Assumes only column 0 has non-zero values
    static std::vector<double> replicateColumn0(const std::vector<double>& vec, int d) {
        std::vector<double> result = vec;
        for (int i = 1; i < d; i *= 2) {
            auto rotated = rotate(result, -i);  // rotate right
            addInPlace(result, rotated);
        }
        return result;
    }

    // SetSlots simulation: change slot count
    // Increase: replicate data (e.g., 64→4096 means 64x replication)
    // Decrease: truncate (assume data is already replicated)
    static std::vector<double> setSlots(const std::vector<double>& vec, int newSlots) {
        int oldSlots = vec.size();
        if (newSlots == oldSlots) return vec;

        if (newSlots > oldSlots) {
            // Increase: replicate
            std::vector<double> result(newSlots, 0.0);
            int copies = newSlots / oldSlots;
            for (int c = 0; c < copies; c++) {
                for (int i = 0; i < oldSlots; i++) {
                    result[c * oldSlots + i] = vec[i];
                }
            }
            return result;
        } else {
            // Decrease: truncate
            return std::vector<double>(vec.begin(), vec.begin() + newSlots);
        }
    }

    // Generate transpose mask for k-th diagonal of d×d matrix
    static std::vector<double> generateTransposeMask(int k, int d) {
        std::vector<double> msk(d * d, 0.0);
        if (k >= 0) {
            for (int j = 0; j < d - k; j++) {
                msk[(d + 1) * j + k] = 1.0;
            }
        } else {
            for (int j = -k; j < d; j++) {
                msk[(d + 1) * j + k] = 1.0;
            }
        }
        return msk;
    }

    // Transpose a d×d matrix packed row-major
    static std::vector<double> transpose(const std::vector<double>& M, int d) {
        std::vector<double> result(d * d, 0.0);

        // k = 0 diagonal
        auto msk = generateTransposeMask(0, d);
        for (int i = 0; i < d * d; i++) {
            result[i] += M[i] * msk[i];
        }

        // Positive diagonals
        for (int k = 1; k < d; k++) {
            msk = generateTransposeMask(k, d);
            auto rotated = rotate(M, (d - 1) * k);
            for (int i = 0; i < d * d; i++) {
                result[i] += rotated[i] * msk[i];
            }
        }

        // Negative diagonals
        for (int k = -1; k > -d; k--) {
            msk = generateTransposeMask(k, d);
            auto rotated = rotate(M, (d - 1) * k);
            for (int i = 0; i < d * d; i++) {
                result[i] += rotated[i] * msk[i];
            }
        }

        return result;
    }

    // Matrix multiplication for d×d matrices (standard, for correctness)
    static std::vector<double> matMult(const std::vector<double>& A,
                                       const std::vector<double>& B,
                                       int d) {
        std::vector<double> C(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int k = 0; k < d; k++) {
                    sum += A[i * d + k] * B[k * d + j];
                }
                C[i * d + j] = sum;
            }
        }
        return C;
    }

    // 8×8 matrix × 8-vector multiplication
    // M is 8×8 row-major (64 elements), v is length 8
    // Returns length 8 result
    static std::vector<double> matVecMult8(const std::vector<double>& M,
                                           const std::vector<double>& v, int dim = 8) {
        std::vector<double> result(dim, 0.0);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                result[i] += M[i * dim + j] * v[j];
            }
        }
        return result;
    }

    // Rebatch: extract f×f submatrix from d×d packed matrix
    // In 64×64 format, the 8×8 result sits at rows 0..7, cols 0..7
    // The d×d matrix has row stride d, we want row stride f
    static std::vector<double> rebatch(const std::vector<double>& M, int d, int f) {
        std::vector<double> result(f * f, 0.0);
        for (int i = 0; i < f; i++) {
            for (int j = 0; j < f; j++) {
                result[i * f + j] = M[i * d + j];
            }
        }
        return result;
    }

    // Invert an f×f matrix using Schulz iteration
    static std::vector<double> invertMatrix(const std::vector<double>& A, int d,
                                            int iterations = 20) {
        std::vector<double> I(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            I[i * d + i] = 1.0;
        }

        double trace = 0.0;
        for (int i = 0; i < d; i++) {
            trace += A[i * d + i];
        }

        if (std::abs(trace) < 1e-10) {
            std::cerr << "Warning: Matrix trace ~ 0, inversion may fail" << std::endl;
            return I;
        }

        // Y_0 = (1/trace(A)) * I
        auto Y = multScalar(I, 1.0 / trace);

        // A_bar = I - A * Y_0
        auto AY = matMult(A, Y, d);
        auto A_bar = sub(I, AY);

        // Iterate: Y = Y * (I + A_bar), A_bar = A_bar^2
        for (int iter = 0; iter < iterations - 1; iter++) {
            auto I_plus_Abar = add(I, A_bar);
            Y = matMult(Y, I_plus_Abar, d);
            A_bar = matMult(A_bar, A_bar, d);
        }

        auto I_plus_Abar = add(I, A_bar);
        Y = matMult(Y, I_plus_Abar, d);

        return Y;
    }

    // Dot product of two vectors
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Debug: print matrix
    static void printMatrix(const std::vector<double>& M, int rows, int cols,
                            const std::string& name, int stride = 0) {
        if (stride == 0) stride = cols;
        std::cout << "=== " << name << " (" << rows << "x" << cols << ") ===" << std::endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed << M[i * stride + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Debug: print vector
    static void printVector(const std::vector<double>& v, int len, const std::string& name) {
        std::cout << "=== " << name << " (len=" << len << ") ===" << std::endl;
        for (int i = 0; i < len && i < (int)v.size(); i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << v[i] << " ";
        }
        std::cout << std::endl << std::endl;
    }
};
