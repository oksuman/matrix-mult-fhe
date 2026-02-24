// Matrix Squaring Benchmark - Naive (d^2 ciphertexts) Algorithm
// Measures: Time, Accuracy, and Memory Usage
// No rotation keys needed - only relinearization key

#include "benchmark_squaring.h"
#include "naive_inversion.h"

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
    parameters.SetBatchSize(1);  // Each ciphertext holds one element

    auto cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    // Only relinearization key needed - NO rotation keys
    cc->EvalMultKeyGen(keyPair.secretKey);

    // Record setup memory (after CryptoContext + relin key only)
    memMetrics.setupMemoryGB = MemoryMonitor::getMemoryUsageGB();

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matOp = std::make_unique<MatrixOperations<d>>(enc, cc, keyPair.publicKey);

    // Print header with parameters
    printSquaringBenchmarkHeader("Naive", d, numRuns, multDepth, Scaling, cc->GetRingDimension(), 1);
    std::cout << "  Ciphertexts: " << d * d << " (d^2)" << std::endl;
    std::cout << "  Rotation keys: None" << std::endl;

    // Generate random matrix
    auto matrix = generateRandomMatrix(d, 42);

    // Encrypt each element separately (d^2 ciphertexts)
    std::vector<Ciphertext<DCRTPoly>> enc_matrix(d * d);
    for (int i = 0; i < d * d; i++) {
        std::vector<double> value = {matrix[i]};
        enc_matrix[i] = enc->encryptInput(value);
    }

    // Measure serialized sizes (once, before timing)
    memMetrics.ciphertextSizeMB = bytesToMB(getCiphertextSize(enc_matrix[0]));
    memMetrics.numCiphertexts = d * d;
    memMetrics.rotationKeysSizeMB = 0.0;  // No rotation keys
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
        std::vector<Ciphertext<DCRTPoly>> current(d * d);
        for (int i = 0; i < d * d; i++) {
            std::vector<double> value = {matrix[i]};
            current[i] = enc->encryptInput(value);
        }

        // Start memory monitor only on first run
        if (run == 0) {
            memMonitor = std::make_unique<MemoryMonitor>(100);
        }

        // ========== Time measurement START ==========
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            current = matOp->matrixMultiply(current, current);
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

        // Decrypt each element and compute error (outside timing)
        std::vector<double> computed(d * d);
        for (int i = 0; i < d * d; i++) {
            Plaintext ptx;
            cc->Decrypt(keyPair.secretKey, current[i], &ptx);
            computed[i] = ptx->GetRealPackedValue()[0];
        }

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
}

int main(int argc, char* argv[]) {
    // Record idle memory at program start
    g_idleMemoryGB = MemoryMonitor::getMemoryUsageGB();

    int numRuns = 1;
    if (argc > 1) {
        numRuns = std::atoi(argv[1]);
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Squaring Benchmark - Naive" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Trials: " << numRuns << std::endl;
    std::cout << "Note: Limited to d<=8 due to d^2 ciphertexts" << std::endl;
    std::cout << "Idle Memory: " << std::fixed << std::setprecision(4) << g_idleMemoryGB << " GB" << std::endl;

    #ifdef _OPENMP
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    // Only d=4,8 due to d^2 ciphertexts complexity
    runSquaringBenchmark<4>(numRuns);
    runSquaringBenchmark<8>(numRuns);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
