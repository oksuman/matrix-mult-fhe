// Matrix Inversion Benchmark - JKLS18 Algorithm
// Measures: Time (seconds) and Accuracy (Frobenius norm, log2 error)

#include "matrix_inversion_algo.h"
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
    int scalarInvIter = getScalarInvIterations(d);
    std::cout << "\n========== JKLS18 Inversion d=" << d << " (scalar_iter=" << scalarInvIter << ") ==========" << std::endl;

    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);

    int batchSize = d * d;
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, batchSize);

    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_JKLS18<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth, scalarInvIter);

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
        auto enc_matrix = enc->encryptInput(matrix);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, result, &ptx);
        ptx->SetLength(d * d);
        std::vector<double> computed = ptx->GetRealPackedValue();
        computed.resize(d * d);

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
    std::cout << "  Matrix Inversion Benchmark - JKLS18" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;

    #ifdef _OPENMP
    // omp_set_num_threads(1);  // Commented for multi-thread quick test
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    runInversionBenchmark<4>(numRuns);
    runInversionBenchmark<8>(numRuns);
    runInversionBenchmark<16>(numRuns);
    runInversionBenchmark<32>(numRuns);
    runInversionBenchmark<64>(numRuns);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
