// Matrix Inversion Benchmark - Naive (d^2 ciphertexts) Algorithm
// Measures: Time (seconds) and Accuracy (Frobenius norm, log2 error)

#include "naive_inversion.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

using namespace lbcrypto;
using namespace BenchmarkConfig;

template <int d>
void runInversionBenchmark(int numRuns = 1) {
    std::cout << "\n========== Naive Inversion d=" << d << " ==========" << std::endl;

    // Unified parameters
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(1);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matOp = std::make_unique<MatrixOperations<d>>(enc, cc, keyPair.publicKey);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Iterations: " << r << std::endl;
    std::cout << "Mult Depth: " << multDepth << std::endl;

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    for (int run = 0; run < numRuns; run++) {
        // Generate random invertible matrix with different seed per run
        std::vector<double> matrix(d * d);
        std::mt19937 gen(42 + run);  // Different seed per trial
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (int i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));

        auto groundTruth = computeGroundTruthInverse(matrix, d);

        // Encrypt each element separately
        std::vector<Ciphertext<DCRTPoly>> enc_matrix(d * d);
        for (int i = 0; i < d * d; i++) {
            std::vector<double> value = {matrix[i]};
            enc_matrix[i] = enc->encryptInput(value);
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto result = matOp->inverseMatrix(enc_matrix, r);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        // Decrypt each element
        std::vector<double> computed(d * d);
        for (int i = 0; i < d * d; i++) {
            Plaintext ptx;
            cc->Decrypt(keyPair.secretKey, result[i], &ptx);
            computed[i] = ptx->GetRealPackedValue()[0];
        }

        ErrorMetrics error;
        error.compute(groundTruth, computed, d);
        if (run == 0) avgError = error;

        std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2)
                  << duration << "s, log2(err)=" << std::setprecision(1) << error.log2FrobError << std::endl;
    }

    double avgTime = totalTime / numRuns;
    double stdDev = 0.0;
    for (double t : times) stdDev += (t - avgTime) * (t - avgTime);
    stdDev = std::sqrt(stdDev / numRuns);

    std::cout << "\n--- Summary (d=" << d << ") ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << avgTime << "s";
    if (numRuns > 1) std::cout << " ± " << stdDev << "s";
    std::cout << std::endl;
    avgError.print();
}

int main(int argc, char* argv[]) {
    int numRuns = 1;
    if (argc > 1) numRuns = std::atoi(argv[1]);

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Inversion Benchmark - Naive" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;
    std::cout << "Note: Limited to d<=8 due to d^2 ciphertexts" << std::endl;

    #ifdef _OPENMP
    // omp_set_num_threads(1);  // Commented for multi-thread quick test
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    runInversionBenchmark<4>(numRuns);
    runInversionBenchmark<8>(numRuns);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
