// Matrix Inversion Benchmark - RT22 Algorithm (Simple: upperbound scaling)
// trace/eval_scalar_inverse 없이 1/d^2 upperbound 로 직접 scaling.
// eval_transpose_scaled 로 transpose+scaling 을 한 번에 처리.
// RT22 는 d=4,8,16,32 만 지원.

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
    std::cout << "\n========== RT22-Simple Inversion d=" << d << " ==========" << std::endl;

    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH_SIMPLE;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);

    int batchSize = d * d * d;
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

    std::vector<int> rotations;
    for (int i = 1; i < d * d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_RT22<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth);

    std::cout << "--- CKKS Parameters ---" << std::endl;
    std::cout << "  multDepth:     " << multDepth << std::endl;
    std::cout << "  scaleModSize:  " << scaleModSize << " bits" << std::endl;
    std::cout << "  firstModSize:  " << firstModSize << " bits" << std::endl;
    std::cout << "  batchSize:     " << batchSize << " (d^3 = " << d << "^3)" << std::endl;
    std::cout << "  ringDimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "  security:      HEStd_128_classic" << std::endl;
    std::cout << "--- Bootstrapping ---" << std::endl;
    std::cout << "  levelBudget:   {" << levelBudget[0] << ", " << levelBudget[1] << "}" << std::endl;
    std::cout << "  bsgsDim:       {" << bsgsDim[0] << ", " << bsgsDim[1] << "}" << std::endl;
    std::cout << "  bootBatchSize: " << d * d << " (d^2)" << std::endl;
    std::cout << "--- Algorithm ---" << std::endl;
    std::cout << "  invIter:       " << r << std::endl;
    std::cout << "  scale:         1/" << d * d << " (no trace/eval_scalar_inverse)" << std::endl;
    std::cout << "  trials:        " << numRuns << std::endl;
    std::cout << "  seed:          1000+run" << std::endl;

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    for (int run = 0; run < numRuns; run++) {
        std::vector<double> matrix(d * d);
        std::mt19937 gen(1000 + run);
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (int i = 0; i < d * d; i++) matrix[i] = dis(gen);
        } while (!utils::isInvertible(matrix, d));

        auto groundTruth = computeGroundTruthInverse(matrix, d);
        auto enc_matrix = enc->encryptInput(matrix);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse_simple(enc_matrix);
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

void runForDimension(int d, int numRuns) {
    switch (d) {
        case 4:  runInversionBenchmark<4>(numRuns);  break;
        case 8:  runInversionBenchmark<8>(numRuns);  break;
        case 16: runInversionBenchmark<16>(numRuns); break;
        case 32: runInversionBenchmark<32>(numRuns); break;
        default:
            std::cerr << "Unsupported dimension for RT22: " << d << std::endl;
            break;
    }
}

int main(int argc, char* argv[]) {
    int numRuns = 1;
    if (argc > 1) numRuns = std::atoi(argv[1]);

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Inversion Benchmark - RT22-Simple" << std::endl;
    std::cout << "  (upperbound scaling, no trace)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;

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
        runInversionBenchmark<16>(numRuns);
        runInversionBenchmark<32>(numRuns);
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
