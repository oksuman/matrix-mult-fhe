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

    // Dimension-specific parameters (no bootstrapping)
    int r = getInversionIterations(d);
    int multDepth;
    if constexpr (d == 4) {
        multDepth = 23;
    } else {
        multDepth = 28;  // d=8
    }
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

    std::cout << "--- CKKS Parameters ---" << std::endl;
    std::cout << "  multDepth:     " << multDepth << std::endl;
    std::cout << "  scaleModSize:  " << scaleModSize << " bits" << std::endl;
    std::cout << "  firstModSize:  " << firstModSize << " bits" << std::endl;
    std::cout << "  batchSize:     1 (one scalar per ciphertext)" << std::endl;
    std::cout << "  ringDimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "  security:      HEStd_128_classic" << std::endl;
    std::cout << "  bootstrapping: None" << std::endl;
    std::cout << "--- Algorithm ---" << std::endl;
    std::cout << "  invIter:       " << r << std::endl;
    std::cout << "  trials:        " << numRuns << std::endl;
    std::cout << "  seed:          1000+run" << std::endl;

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    for (int run = 0; run < numRuns; run++) {
        // Generate random invertible matrix with different seed per run
        std::vector<double> matrix(d * d);
        std::mt19937 gen(1000 + run);  // Different seed per trial
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

        // Print final ciphertext level
        int finalLevel = result[0]->GetLevel();
        std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2)
                  << duration << "s, log2(err)=" << std::setprecision(1) << error.log2FrobError
                  << ", final_level=" << finalLevel << "/" << multDepth << std::endl;
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

void runForDimension(int d, int numRuns) {
    switch (d) {
        case 4:  runInversionBenchmark<4>(numRuns);  break;
        case 8:  runInversionBenchmark<8>(numRuns);  break;
        case 16: runInversionBenchmark<16>(numRuns); break;
        default:
            std::cerr << "Unsupported dimension for Naive: " << d << " (max d=16)" << std::endl;
            break;
    }
}

int main(int argc, char* argv[]) {
    int numRuns = 1;
    if (argc > 1) numRuns = std::atoi(argv[1]);

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Inversion Benchmark - Naive" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;
    std::cout << "Note: Limited to d<=16 due to d^2 ciphertexts" << std::endl;

    #ifdef _OPENMP
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    if (argc > 2) {
        int d = std::atoi(argv[2]);
        runForDimension(d, numRuns);
    } else {
        runInversionBenchmark<4>(numRuns);
        runInversionBenchmark<8>(numRuns);
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
