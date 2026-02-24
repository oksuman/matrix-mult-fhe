// Deep Multiplication (Squaring) Benchmark Header
// Measures: Time (seconds), Accuracy (Frobenius norm), and Memory Usage
// Format: Same as inversion benchmarks
#pragma once

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "matrix_algo_multiPack.h"
#include "diagonal_packing.h"
#include "../benchmark_config.h"
#include "../memory_tracker.h"

#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;
using namespace BenchmarkConfig;
using namespace MemoryUtils;

// Number of squaring iterations
constexpr int SQUARING_ITERATIONS = 15;
constexpr int Scaling = 50;

// Plaintext matrix multiplication for ground truth
inline std::vector<double> multiplyMatricesPlaintext(const std::vector<double>& A,
                                                      const std::vector<double>& B, int d) {
    std::vector<double> C(d * d, 0.0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                C[i * d + j] += A[i * d + k] * B[k * d + j];
            }
        }
    }
    return C;
}

// Compute A^(2^n) in plaintext for ground truth
inline std::vector<double> computeGroundTruthSquaring(const std::vector<double>& matrix,
                                                       int d, int iterations) {
    std::vector<double> result = matrix;
    for (int i = 0; i < iterations; i++) {
        result = multiplyMatricesPlaintext(result, result, d);
    }
    return result;
}

// Generate random matrix with small entries to prevent overflow during squaring
inline std::vector<double> generateRandomMatrix(int d, unsigned seed = 42) {
    std::vector<double> matrix(d * d);
    std::mt19937 gen(seed);
    // Use small values to prevent overflow after 15 squarings
    // ||A^(2^15)|| can grow exponentially, so we use entries around 1/sqrt(d)
    double scale = 0.3 / std::sqrt((double)d);
    std::uniform_real_distribution<double> dis(-scale, scale);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// Print benchmark header with parameters
inline void printSquaringBenchmarkHeader(const std::string& algorithmName, int d, int numRuns,
                                          int multDepth, int scalingMod, int ringDim, int batchSize) {
    std::cout << "\n========== " << algorithmName << " d=" << d << " ==========" << std::endl;
    std::cout << "--- CKKS Parameters ---" << std::endl;
    std::cout << "  multDepth:     " << multDepth
              << " (" << SQUARING_ITERATIONS << " iter x "
              << multDepth / SQUARING_ITERATIONS << " levels/iter)" << std::endl;
    std::cout << "  scaleModSize:  " << scalingMod << " bits" << std::endl;
    std::cout << "  batchSize:     " << batchSize << std::endl;
    std::cout << "  ringDimension: " << ringDim << std::endl;
    std::cout << "  security:      HEStd_128_classic" << std::endl;
    std::cout << "  bootstrapping: None" << std::endl;
    std::cout << "--- Experiment ---" << std::endl;
    std::cout << "  squaringIter:  " << SQUARING_ITERATIONS << std::endl;
    std::cout << "  trials:        " << numRuns << std::endl;
    std::cout << "  seed:          42 (fixed)" << std::endl;
}
