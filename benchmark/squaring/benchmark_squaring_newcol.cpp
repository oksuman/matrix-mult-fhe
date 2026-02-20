// Matrix Squaring Benchmark - NewCol Algorithm
// Measures: Time, Accuracy, and Memory Usage

#include "benchmark_squaring.h"

using namespace lbcrypto;
using namespace BenchmarkConfig;
using namespace MemoryUtils;

// Global idle memory (recorded at program start)
static double g_idleMemoryGB = 0.0;

template <int d>
void runSquaringBenchmark(int numRuns = 1) {
    MemoryMetrics memMetrics;
    memMetrics.idleMemoryGB = g_idleMemoryGB;

    // Setup encryption
    const int multDepth = SQUARING_ITERATIONS * 2;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(Scaling);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetBatchSize(d * d);

    auto cc = GenCryptoContext(parameters);
    int max_batch = cc->GetRingDimension() / 2;

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    int s = std::min(max_batch / d / d, d);
    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    // Record setup memory (after CryptoContext + all keys)
    memMetrics.setupMemoryGB = MemoryMonitor::getMemoryUsageGB();

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

    // Print header with parameters
    printSquaringBenchmarkHeader("NewCol", d, numRuns, multDepth, Scaling, cc->GetRingDimension());

    // Generate random matrix
    auto matrix = generateRandomMatrix(d, 42);
    auto enc_matrix = enc->encryptInput(matrix);

    // Measure serialized sizes (once, before timing)
    memMetrics.ciphertextSizeMB = bytesToMB(getCiphertextSize(enc_matrix));
    memMetrics.numCiphertexts = 1;
    memMetrics.rotationKeysSizeMB = bytesToMB(getRotationKeysSize(cc));
    memMetrics.relinKeySizeMB = bytesToMB(getRelinKeySize(cc));

    // Compute ground truth
    auto groundTruth = computeGroundTruthSquaring(matrix, d, SQUARING_ITERATIONS);

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    // Memory monitor for first run only
    std::unique_ptr<MemoryMonitor> memMonitor;

    for (int run = 0; run < numRuns; run++) {
        // Fresh encryption for each run
        auto current = enc->encryptInput(matrix);

        // Start memory monitor only on first run
        if (run == 0) {
            memMonitor = std::make_unique<MemoryMonitor>(100);
        }

        // ========== Time measurement START ==========
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            current = algo->eval_mult(current, current);
        }

        auto end = std::chrono::high_resolution_clock::now();
        // ========== Time measurement END ==========

        // Record peak memory (first run only)
        if (run == 0 && memMonitor) {
            memMetrics.peakMemoryGB = memMonitor->getPeakMemoryGB();
            memMetrics.avgMemoryGB = memMonitor->getAverageMemoryGB();
            memMonitor.reset();  // Stop monitoring
        }

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        // Decrypt and compute error (outside timing)
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, current, &ptx);
        ptx->SetLength(d * d);
        std::vector<double> computed = ptx->GetRealPackedValue();
        computed.resize(d * d);

        ErrorMetrics error;
        error.compute(groundTruth, computed, d);

        if (run == 0) {
            avgError = error;
        }

        std::cout << "  [" << (run + 1) << "/" << numRuns << "] "
                  << std::fixed << std::setprecision(2) << duration << "s"
                  << ", log2(err)=" << std::setprecision(1) << error.log2FrobError << std::endl;
    }

    // Summary
    double avgTime = totalTime / numRuns;
    double stdDev = 0.0;
    for (double t : times) {
        stdDev += (t - avgTime) * (t - avgTime);
    }
    stdDev = std::sqrt(stdDev / numRuns);

    std::cout << "\n--- Summary (d=" << d << ") ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << avgTime << "s";
    if (numRuns > 1) {
        std::cout << " +/- " << stdDev << "s";
    }
    std::cout << std::endl;
    avgError.print();
    memMetrics.print();

    // Cleanup to free memory before next dimension
    cc->ClearEvalMultKeys();
    cc->ClearEvalAutomorphismKeys();
}

int main(int argc, char* argv[]) {
    // Record idle memory at program start
    g_idleMemoryGB = MemoryMonitor::getMemoryUsageGB();

    int numRuns = 1;
    if (argc > 1) {
        numRuns = std::atoi(argv[1]);
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Squaring Benchmark - NewCol" << std::endl;
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
