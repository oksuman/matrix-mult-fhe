// Matrix Squaring Benchmark - Diagonal Packing Algorithm
// Measures: Time, Accuracy, and Memory Usage
// Note: Uses d ciphertexts (one per diagonal)

#include "benchmark_squaring.h"

using namespace lbcrypto;
using namespace BenchmarkConfig;
using namespace MemoryUtils;

static double g_idleMemoryGB = 0.0;

template <int d>
void runSquaringBenchmark(int numRuns = 1) {
    MemoryMetrics memMetrics;
    memMetrics.idleMemoryGB = g_idleMemoryGB;

    const int multDepth = SQUARING_ITERATIONS;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(Scaling);
    parameters.SetBatchSize(d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    memMetrics.setupMemoryGB = MemoryMonitor::getMemoryUsageGB();

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixInv_diag<d>>(enc, cc, keyPair.publicKey, rotations, -1);

    printSquaringBenchmarkHeader("Diagonal", d, numRuns, multDepth, Scaling, cc->GetRingDimension());
    std::cout << "  Ciphertexts: " << d << std::endl;

    auto matrix = generateRandomMatrix(d, 42);
    auto groundTruth = computeGroundTruthSquaring(matrix, d, SQUARING_ITERATIONS);

    auto diagonals = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto& diag : diagonals) {
        enc_matrix.push_back(enc->encryptInput(diag));
    }

    memMetrics.ciphertextSizeMB = bytesToMB(getCiphertextSize(enc_matrix[0]));
    memMetrics.numCiphertexts = d;
    memMetrics.rotationKeysSizeMB = bytesToMB(getRotationKeysSize(cc));
    memMetrics.relinKeySizeMB = bytesToMB(getRelinKeySize(cc));

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;
    std::unique_ptr<MemoryMonitor> memMonitor;

    for (int run = 0; run < numRuns; run++) {
        std::vector<Ciphertext<DCRTPoly>> current;
        for (const auto& diag : diagonals) {
            current.push_back(enc->encryptInput(diag));
        }

        if (run == 0) {
            memMonitor = std::make_unique<MemoryMonitor>(100);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            current = algo->eval_mult(current, current);
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (run == 0 && memMonitor) {
            memMetrics.peakMemoryGB = memMonitor->getPeakMemoryGB();
            memMetrics.avgMemoryGB = memMonitor->getAverageMemoryGB();
            memMonitor.reset();
        }

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        std::vector<double> computed(d * d, 0.0);
        for (int diagIdx = 0; diagIdx < d; diagIdx++) {
            Plaintext ptx;
            cc->Decrypt(keyPair.secretKey, current[diagIdx], &ptx);
            ptx->SetLength(d);
            auto diagVec = ptx->GetRealPackedValue();
            for (int i = 0; i < d; i++) {
                int j = (i + diagIdx) % d;
                computed[i * d + j] = diagVec[i];
            }
        }

        ErrorMetrics error;
        error.compute(groundTruth, computed, d);
        if (run == 0) avgError = error;

        std::cout << "  [" << (run + 1) << "/" << numRuns << "] "
                  << std::fixed << std::setprecision(2) << duration << "s"
                  << ", log2(err)=" << std::setprecision(1) << error.log2FrobError << std::endl;
    }

    double avgTime = totalTime / numRuns;
    double stdDev = 0.0;
    for (double t : times) stdDev += (t - avgTime) * (t - avgTime);
    stdDev = std::sqrt(stdDev / numRuns);

    std::cout << "\n--- Summary (d=" << d << ") ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << avgTime << "s";
    if (numRuns > 1) std::cout << " +/- " << stdDev << "s";
    std::cout << std::endl;
    avgError.print();
    memMetrics.print();

    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
}

int main(int argc, char* argv[]) {
    g_idleMemoryGB = MemoryMonitor::getMemoryUsageGB();
    int numRuns = (argc > 1) ? std::atoi(argv[1]) : 1;

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Squaring Benchmark - Diagonal" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Trials: " << numRuns << std::endl;
    std::cout << "Idle Memory: " << std::fixed << std::setprecision(4) << g_idleMemoryGB << " GB" << std::endl;

    #ifdef _OPENMP
    omp_set_num_threads(1);
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    runSquaringBenchmark<4>(numRuns);
    runSquaringBenchmark<8>(numRuns);
    runSquaringBenchmark<16>(numRuns);
    runSquaringBenchmark<32>(numRuns);
    runSquaringBenchmark<64>(numRuns);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
