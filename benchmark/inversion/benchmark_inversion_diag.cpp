// Matrix Inversion Benchmark - Diagonal Packing Algorithm
// Measures: Time (seconds) and Accuracy (Frobenius norm, log2 error)

#include "matrix_algo_multiPack.h"
#include "diagonal_packing.h"
#include "matrix_utils.h"
#include "encryption.h"
#include "rotation.h"
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
    std::cout << "\n========== Diagonal Inversion d=" << d << " ==========" << std::endl;

    int r = getInversionIterations(d);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(r + 9);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInv_diag<d>>(
        enc, cc, keyPair.publicKey, rotations, r);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Iterations: " << r << std::endl;

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1, 1);

    // Generate diagonal dominant matrix for stability
    do {
        for (size_t i = 0; i < d; i++) {
            double sum = 0;
            for (size_t j = 0; j < d; j++) {
                if (i != j) {
                    matrix[i * d + j] = dis(gen) * 0.1;
                    sum += std::abs(matrix[i * d + j]);
                }
            }
            matrix[i * d + i] = sum + std::abs(dis(gen)) + 0.5;
        }
    } while (!utils::isDiagonalMatrixInvertible(matrix, d));

    auto groundTruth = computeGroundTruthInverse(matrix, d);

    auto diagonals = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto& diag : diagonals) {
        enc_matrix.push_back(enc->encryptInput(diag));
    }

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    for (int run = 0; run < numRuns; run++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        // Reconstruct matrix from diagonal form
        std::vector<double> computed(d * d, 0.0);
        for (int k = 0; k < d; k++) {
            Plaintext ptx;
            cc->Decrypt(keyPair.secretKey, result[k], &ptx);
            ptx->SetLength(d);
            auto diagVec = ptx->GetRealPackedValue();
            for (int i = 0; i < d; i++) {
                int j = (i + k) % d;
                computed[i * d + j] = diagVec[i];
            }
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
    std::cout << "  Matrix Inversion Benchmark - Diagonal" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;

    #ifdef _OPENMP
    omp_set_num_threads(1);
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    runInversionBenchmark<4>(numRuns);
    runInversionBenchmark<8>(numRuns);
    runInversionBenchmark<16>(numRuns);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
