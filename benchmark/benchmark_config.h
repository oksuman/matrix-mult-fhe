#pragma once

#include <openfhe.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

// ============================================================
// Unified Benchmark Configuration
// ============================================================

namespace BenchmarkConfig {

// Encryption Parameters (unified for all benchmarks)
constexpr int MULT_DEPTH = 30;
constexpr int FIRST_MOD_SIZE = 60;
constexpr int SCALE_MOD_SIZE = 59;
// LEVEL_BUDGET: use {4, 4} when creating std::vector

// Matrix Inversion Iterations by Dimension (95th percentile)
constexpr int getInversionIterations(int d) {
    switch(d) {
        case 4:  return 18;
        case 8:  return 22;
        case 16: return 25;
        case 32: return 27;
        case 64: return 30;
        default: return 25;
    }
}

// Scalar Inversion Iterations: 1 for all
constexpr int getScalarInvIterations(int d) {
    return 1;
}

// Benchmark Execution
constexpr int NUM_RUNS = 10;  // Number of runs for averaging

// Deep Multiplication
constexpr int DEEP_MULT_ITERATIONS = 10;  // Squaring iterations

// Naive benchmark dimension limit
constexpr int NAIVE_MAX_DIM = 8;

// Print current thread count
inline void printThreadInfo() {
    #ifdef _OPENMP
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "Single Thread (OpenMP not enabled)" << std::endl;
    #endif
}

// Set single thread execution
inline void setSingleThread() {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif
    // Also set OpenFHE's internal threading if available
    // Note: OpenFHE uses SetNumThreads from its own API
}

// Upper Bound Estimation Functions
inline double getMatrixInversionUpperBound(int d) {
    // For random matrix with entries in [-1, 1]:
    // trace(M^T M) <= d^2 (conservative upper bound)
    return static_cast<double>(d * d);
}

inline double getLinearRegressionUpperBound(int numSamples, int numFeatures) {
    // For min-max normalized [0,1] data:
    // trace(X^T X) <= N * d
    return static_cast<double>(numSamples * numFeatures);
}

inline double getLDAUpperBound(int numSamples, int numFeatures) {
    // For Z-score normalized data:
    // trace(S_W) <= N * d
    return static_cast<double>(numSamples * numFeatures);
}

inline double getFixedHessianUpperBound(int numSamples, int numFeatures) {
    // Hessian H = X^T W X where W = diag(p(1-p)), p(1-p) <= 0.25
    // trace(H) <= 0.25 * N * d, but use N * d for safety
    return static_cast<double>(numSamples * numFeatures);
}

// Error Metrics for Accuracy Measurement
struct ErrorMetrics {
    double frobeniusNorm;      // ||A - B||_F
    double maxNorm;            // max|A_ij - B_ij|
    double relativeFrobenius;  // ||A - B||_F / ||A||_F
    double relativeMax;        // max|A_ij - B_ij| / max|A_ij|
    double log2FrobError;      // log2(relativeFrobenius)
    double log2MaxError;       // log2(relativeMax)

    void compute(const std::vector<double>& groundTruth,
                 const std::vector<double>& computed,
                 int d) {
        double sumSqDiff = 0.0, sumSqGT = 0.0;
        double maxDiff = 0.0, maxGT = 0.0;

        for (int i = 0; i < d * d; i++) {
            double diff = std::abs(groundTruth[i] - computed[i]);
            double gt = std::abs(groundTruth[i]);

            sumSqDiff += diff * diff;
            sumSqGT += gt * gt;
            maxDiff = std::max(maxDiff, diff);
            maxGT = std::max(maxGT, gt);
        }

        frobeniusNorm = std::sqrt(sumSqDiff);
        maxNorm = maxDiff;

        double frobGT = std::sqrt(sumSqGT);
        relativeFrobenius = (frobGT > 1e-15) ? frobeniusNorm / frobGT : frobeniusNorm;
        relativeMax = (maxGT > 1e-15) ? maxNorm / maxGT : maxNorm;

        log2FrobError = (relativeFrobenius > 1e-15) ? std::log2(relativeFrobenius) : -50.0;
        log2MaxError = (relativeMax > 1e-15) ? std::log2(relativeMax) : -50.0;
    }

    void print(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(6);
        os << "  Frobenius Norm Error: " << frobeniusNorm << std::endl;
        os << "  Max Norm Error:       " << maxNorm << std::endl;
        os << "  Relative Frobenius:   " << relativeFrobenius << std::endl;
        os << "  Relative Max:         " << relativeMax << std::endl;
        os << "  log2(Rel. Frob.):     " << log2FrobError << std::endl;
        os << "  log2(Rel. Max):       " << log2MaxError << std::endl;
    }
};

// Result Structure for Benchmarks
struct BenchmarkResult {
    std::string algorithm;
    int dimension;
    int iterations;
    double avgTime;      // Average time in seconds
    double stdDev;       // Standard deviation
    ErrorMetrics error;

    void printSummary(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(2);
        os << algorithm << " (d=" << dimension << "): ";
        os << avgTime << "s ± " << stdDev << "s, ";
        os << "log2(err)=" << std::setprecision(1) << error.log2FrobError << std::endl;
    }
};

// Gauss-Jordan Elimination for Ground Truth Inverse
inline std::vector<double> computeGroundTruthInverse(const std::vector<double>& matrix, int d) {
    std::vector<double> A = matrix;
    std::vector<double> I(d * d, 0.0);
    for (int i = 0; i < d; i++) I[i * d + i] = 1.0;

    // Gauss-Jordan elimination
    for (int col = 0; col < d; col++) {
        // Find pivot
        int maxRow = col;
        double maxVal = std::abs(A[col * d + col]);
        for (int row = col + 1; row < d; row++) {
            if (std::abs(A[row * d + col]) > maxVal) {
                maxVal = std::abs(A[row * d + col]);
                maxRow = row;
            }
        }

        // Swap rows
        if (maxRow != col) {
            for (int j = 0; j < d; j++) {
                std::swap(A[col * d + j], A[maxRow * d + j]);
                std::swap(I[col * d + j], I[maxRow * d + j]);
            }
        }

        // Scale pivot row
        double pivot = A[col * d + col];
        if (std::abs(pivot) < 1e-12) continue;

        for (int j = 0; j < d; j++) {
            A[col * d + j] /= pivot;
            I[col * d + j] /= pivot;
        }

        // Eliminate column
        for (int row = 0; row < d; row++) {
            if (row == col) continue;
            double factor = A[row * d + col];
            for (int j = 0; j < d; j++) {
                A[row * d + j] -= factor * A[col * d + j];
                I[row * d + j] -= factor * I[col * d + j];
            }
        }
    }

    return I;
}

}  // namespace BenchmarkConfig
