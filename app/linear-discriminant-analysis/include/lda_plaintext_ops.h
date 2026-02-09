#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

// CKKS-friendly plaintext operations
// These simulate SIMD operations that would be performed on encrypted data

class PlaintextOps {
public:
    // Rotate vector left by rotateIndex (positive = left, negative = right)
    // In CKKS, this corresponds to EvalRotate
    static std::vector<double> rotate(const std::vector<double>& vec, int rotateIndex) {
        if (vec.empty()) return vec;

        std::vector<double> result = vec;
        int n = result.size();

        // Normalize rotation index
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

    // In-place addition
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

    // Apply mask (multiply element-wise with mask vector)
    static std::vector<double> mask(const std::vector<double>& vec, const std::vector<double>& maskVec) {
        return mult(vec, maskVec);
    }

    // Folding sum: sum rows to get a single row replicated
    // Given vector of length s̃ * f̃, fold to get sum of all s̃ rows
    // Each row has f̃ elements
    // Result: the sum is replicated s̃ times
    static std::vector<double> foldingSum(const std::vector<double>& vec,
                                          size_t paddedSamples,
                                          size_t paddedFeatures) {
        std::vector<double> result = vec;

        // log2(paddedSamples) rotations and additions
        for (size_t i = 1; i < paddedSamples; i *= 2) {
            auto rotated = rotate(result, i * paddedFeatures);
            addInPlace(result, rotated);
        }

        return result;
    }

    // Compute mean by folding sum and dividing by count
    // Returns replicated form (mean vector repeated paddedSamples times)
    static std::vector<double> computeMean(const std::vector<double>& vec,
                                           size_t actualSamples,
                                           size_t paddedSamples,
                                           size_t paddedFeatures) {
        auto summed = foldingSum(vec, paddedSamples, paddedFeatures);
        return multScalar(summed, 1.0 / actualSamples);
    }

    // Generate transpose mask for k-th diagonal
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

    // Transpose a d x d matrix packed in row-major order
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

    // Matrix multiplication for d x d matrices (single-pack style)
    // Uses diagonal extraction method similar to CKKS matrix mult
    static std::vector<double> matMult(const std::vector<double>& A,
                                       const std::vector<double>& B,
                                       int d) {
        std::vector<double> C(d * d, 0.0);

        for (int k = 0; k < d; k++) {
            // Extract k-th diagonal of A (with rotation)
            std::vector<double> diagA(d * d, 0.0);
            for (int i = 0; i < d; i++) {
                int j = (i + k) % d;
                diagA[i * d + j] = A[i * d + j];
            }

            // Rotate B's columns
            std::vector<double> rotB = rotate(B, k);

            // Hadamard product and accumulate
            for (int i = 0; i < d * d; i++) {
                // We need to think about this differently
                // Let's use the standard algorithm but in SIMD style
            }
        }

        // Actually, let's implement it more directly for correctness
        // Standard matrix multiplication (SIMD-friendly form)
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

    // Matrix multiplication for rectangular matrices stored in padded form
    // A is m x n (stored in paddedM x paddedN), B is n x p
    // Actually for LDA we need X^T * X where X is s x f
    // So we compute (s x f)^T * (s x f) = (f x s) * (s x f) = f x f

    // Compute X^T * X for matrix X (samples x features)
    // X is stored as paddedSamples rows, each of paddedFeatures
    static std::vector<double> computeXtX(const std::vector<double>& X,
                                          size_t actualSamples,
                                          size_t actualFeatures,
                                          size_t paddedSamples,
                                          size_t paddedFeatures) {
        // Result is actualFeatures x actualFeatures, padded to paddedFeatures x paddedFeatures
        std::vector<double> XtX(paddedFeatures * paddedFeatures, 0.0);

        // X^T * X = sum over samples of outer product
        for (size_t s = 0; s < actualSamples; s++) {
            for (size_t i = 0; i < actualFeatures; i++) {
                for (size_t j = 0; j < actualFeatures; j++) {
                    XtX[i * paddedFeatures + j] += X[s * paddedFeatures + i] * X[s * paddedFeatures + j];
                }
            }
        }

        return XtX;
    }

    // Compute outer product of two vectors (both length f, padded to paddedF)
    // Result is f x f matrix (padded to paddedF x paddedF)
    static std::vector<double> outerProduct(const std::vector<double>& a,
                                            const std::vector<double>& b,
                                            size_t actualF,
                                            size_t paddedF) {
        std::vector<double> result(paddedF * paddedF, 0.0);

        for (size_t i = 0; i < actualF; i++) {
            for (size_t j = 0; j < actualF; j++) {
                result[i * paddedF + j] = a[i] * b[j];
            }
        }

        return result;
    }

    // Extract first row from replicated mean vector
    static std::vector<double> extractFirstRow(const std::vector<double>& vec, size_t paddedFeatures) {
        std::vector<double> result(paddedFeatures, 0.0);
        for (size_t i = 0; i < paddedFeatures; i++) {
            result[i] = vec[i];
        }
        return result;
    }

    // Replicate a row vector to fill s̃ rows
    static std::vector<double> replicateRow(const std::vector<double>& row,
                                            size_t targetPaddedSamples,
                                            size_t paddedFeatures) {
        std::vector<double> result(targetPaddedSamples * paddedFeatures, 0.0);

        for (size_t s = 0; s < targetPaddedSamples; s++) {
            for (size_t f = 0; f < paddedFeatures; f++) {
                result[s * paddedFeatures + f] = row[f];
            }
        }

        return result;
    }

    // Debug: print matrix
    // stride: row stride in storage (if 0, use cols as stride)
    static void printMatrix(const std::vector<double>& M, int rows, int cols, const std::string& name, int stride = 0) {
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
        for (int i = 0; i < len; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << v[i] << " ";
        }
        std::cout << std::endl << std::endl;
    }
};
