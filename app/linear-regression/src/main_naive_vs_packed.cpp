// main_naive_vs_packed.cpp
// Linear Regression: Naive (entry-level) vs Packed (AR24, NewCol) comparison
// Each algorithm uses its own CryptoContext for fair comparison

#include "lr_naive.h"
#include "lr_ar24.h"
#include "lr_newcol.h"
#include "../../common/evaluation_metrics.h"
#include "memory_tracker.h"
#include "encryption.h"
#include "rotation.h"
#include "csv_processor.h"
#include "mse_calculator.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

// Result structure
struct AlgorithmResult {
    std::string name;
    EvalMetrics::RegressionResult regResult;
    std::chrono::duration<double> step1{0};  // Precomputation (XtX + Xty)
    std::chrono::duration<double> step2{0};  // Matrix inversion
    std::chrono::duration<double> step3{0};  // Weight computation
    std::chrono::duration<double> total{0};
    MemoryUtils::MemoryMetrics mem;
    int ctCount = 0;
    bool valid = false;
};

// Run Naive algorithm in its own scope
AlgorithmResult runNaive(const std::string& trainFile,
                          const std::string& testFile,
                          double idleMemGB, bool verbose) {
    using namespace std::chrono;
    AlgorithmResult result;
    result.name = "Naive";

    std::cout << "\n" << std::string(64, '=') << std::endl;
    std::cout << "  Naive (Entry-Level Encryption)" << std::endl;
    std::cout << std::string(64, '=') << std::endl;

    // Load training data
    std::vector<double> features, outcomes;
    CSVProcessor::processDataset(trainFile, features, outcomes,
                                 LinearRegression_Naive::D, LinearRegression_Naive::N);

    // Setup
    LinearRegression_Naive naive;
    naive.setVerbose(verbose);

    std::cout << "  Setting up CryptoContext (batchSize=1, multDepth="
              << LinearRegression_Naive::MULT_DEPTH << ")..." << std::endl;
    naive.setup();

    double setupMemGB = MemoryMonitor::getMemoryUsageGB();
    result.mem.idleMemoryGB = idleMemGB;
    result.mem.setupMemoryGB = setupMemGB;

    // Measure serialized sizes
    result.mem.ciphertextSizeMB = MemoryUtils::bytesToMB(
        MemoryUtils::getCiphertextSize(naive.getSampleCiphertext()));
    result.mem.rotationKeysSizeMB = 0.0;  // No rotation keys
    result.mem.relinKeySizeMB = MemoryUtils::bytesToMB(
        MemoryUtils::getRelinKeySize(naive.getCC()));

    std::cout << "  Setup complete. Memory: " << std::fixed << std::setprecision(4)
              << (setupMemGB - idleMemGB) << " GB overhead" << std::endl;

    // Run computation with memory monitoring
    MemoryMonitor monitor(200);
    auto totalStart = high_resolution_clock::now();

    // Step 1: Precomputation (XtX + Xty)
    std::cout << "\n  [Step 1] Precomputation (row-by-row XtX + Xty)..." << std::endl;
    auto [step1_time, ctCount] = naive.computePrecomputation(features, outcomes);
    result.step1 = step1_time;
    result.ctCount = ctCount;
    std::cout << "  Step 1 time: " << std::fixed << std::setprecision(2)
              << step1_time.count() << " s (CT count: " << ctCount << ")" << std::endl;

    // Step 2: Matrix Inversion
    std::cout << "  [Step 2] Matrix Inversion (r=" << LinearRegression_Naive::INV_ITER
              << ", no bootstrapping)..." << std::endl;
    result.step2 = naive.computeInverse();
    std::cout << "  Step 2 time: " << std::fixed << std::setprecision(2)
              << result.step2.count() << " s" << std::endl;

    // Step 3: Weight Computation
    std::cout << "  [Step 3] Weight Computation (entry-level mat-vec)..." << std::endl;
    result.step3 = naive.computeWeights();
    std::cout << "  Step 3 time: " << std::fixed << std::setprecision(2)
              << result.step3.count() << " s" << std::endl;

    auto totalEnd = high_resolution_clock::now();
    result.total = totalEnd - totalStart;
    result.mem.peakMemoryGB = monitor.getPeakMemoryGB();
    result.mem.avgMemoryGB = monitor.getAverageMemoryGB();
    result.mem.numCiphertexts = ctCount;

    // Inference
    auto weights = naive.getDecryptedWeights();
    std::cout << "\n  Weights: ";
    for (int j = 0; j < LinearRegression_Naive::D; j++) {
        std::cout << std::setw(10) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << std::endl;

    // Compute MSE
    std::vector<double> test_features, test_outcomes;
    CSVProcessor::processDataset(testFile, test_features, test_outcomes,
                                 LinearRegression_Naive::D, LinearRegression_Naive::N);

    std::vector<double> predictions(LinearRegression_Naive::N);
    std::vector<double> groundTruth(LinearRegression_Naive::N);
    for (int i = 0; i < LinearRegression_Naive::N; i++) {
        double pred = 0.0;
        for (int j = 0; j < LinearRegression_Naive::D; j++) {
            pred += test_features[i * LinearRegression_Naive::N + j] * weights[j];
        }
        predictions[i] = pred;
        groundTruth[i] = test_outcomes[i];
    }
    result.regResult.compute(predictions, groundTruth);
    std::cout << "  MSE: " << std::setprecision(6) << result.regResult.mse << std::endl;

    result.valid = true;
    return result;
}

// Run a packed algorithm (AR24 or NewCol) with shared CC
template<typename LRAlgorithm>
AlgorithmResult runPacked(const std::string& algorithmName,
                           const std::string& trainFile,
                           const std::string& testFile,
                           double idleMemGB, bool verbose) {
    using namespace std::chrono;
    AlgorithmResult result;
    result.name = algorithmName;

    std::cout << "\n" << std::string(64, '=') << std::endl;
    std::cout << "  " << algorithmName << " (Matrix-Packed Encryption)" << std::endl;
    std::cout << std::string(64, '=') << std::endl;

    // Setup CKKS context
    int multDepth = 30;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(SAMPLE_DIM * SAMPLE_DIM);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < 1 << 16; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    std::cout << "  Generating rotation keys..." << std::flush;
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << " Done." << std::endl;

    std::cout << "  Setting up bootstrapping..." << std::flush;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, FEATURE_DIM * FEATURE_DIM);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, FEATURE_DIM * FEATURE_DIM);
    std::cout << " Done." << std::endl;

    double setupMemGB = MemoryMonitor::getMemoryUsageGB();
    result.mem.idleMemoryGB = idleMemGB;
    result.mem.setupMemoryGB = setupMemGB;

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

    // Load and encrypt data
    std::vector<double> features, outcomes;
    CSVProcessor::processDataset(trainFile, features, outcomes,
                                 FEATURE_DIM, SAMPLE_DIM);

    auto X = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(features, 1, 0, nullptr, SAMPLE_DIM * SAMPLE_DIM));
    auto y = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(outcomes, 1, 0, nullptr, SAMPLE_DIM));

    // Measure sizes
    result.mem.ciphertextSizeMB = MemoryUtils::bytesToMB(MemoryUtils::getCiphertextSize(X));
    result.mem.rotationKeysSizeMB = MemoryUtils::bytesToMB(MemoryUtils::getRotationKeysSize(cc));
    result.mem.relinKeySizeMB = MemoryUtils::bytesToMB(MemoryUtils::getRelinKeySize(cc));
    result.ctCount = 2;  // X + y
    result.mem.numCiphertexts = 2;

    // Run with memory monitoring
    MemoryMonitor monitor(200);
    auto totalStart = high_resolution_clock::now();

    LRAlgorithm lr(enc, cc, keyPair, rotations, multDepth);
    lr.setVerbose(verbose);

    auto [s1, s2, s3, s4] = lr.trainWithTimings(X, y);

    auto totalEnd = high_resolution_clock::now();
    result.total = totalEnd - totalStart;
    result.step1 = s1 + s3;  // XtX + Xty combined
    result.step2 = s2;       // Inversion
    result.step3 = s4;       // Weight computation
    result.mem.peakMemoryGB = monitor.getPeakMemoryGB();
    result.mem.avgMemoryGB = monitor.getAverageMemoryGB();

    // Inference
    Plaintext ptx;
    cc->Decrypt(keyPair.secretKey, lr.getWeights(), &ptx);
    ptx->SetLength(FEATURE_DIM * FEATURE_DIM);
    auto weights = ptx->GetRealPackedValue();

    std::cout << "\n  Weights: ";
    for (int j = 0; j < std::min(FEATURE_DIM, 8); j++) {
        std::cout << std::setw(10) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << std::endl;

    // Compute MSE
    std::vector<double> test_features, test_outcomes;
    CSVProcessor::processDataset(testFile, test_features, test_outcomes,
                                 FEATURE_DIM, SAMPLE_DIM);

    std::vector<double> predictions(SAMPLE_DIM);
    std::vector<double> groundTruth(SAMPLE_DIM);
    for (int i = 0; i < SAMPLE_DIM; i++) {
        double pred = 0.0;
        for (int j = 0; j < FEATURE_DIM; j++) {
            pred += test_features[i * SAMPLE_DIM + j] * weights[j];
        }
        predictions[i] = pred;
        groundTruth[i] = test_outcomes[i];
    }
    result.regResult.compute(predictions, groundTruth);
    std::cout << "  MSE: " << std::setprecision(6) << result.regResult.mse << std::endl;

    result.valid = true;
    return result;
}

void printComparisonTable(const std::vector<AlgorithmResult>& results) {
    std::cout << "\n";
    std::cout << "================================================================" << std::endl;
    std::cout << "      Linear Regression -- Naive vs Packed Comparison" << std::endl;
    std::cout << "      Samples: " << LinearRegression_Naive::N << " train / "
              << LinearRegression_Naive::N << " test, Features: "
              << LinearRegression_Naive::D << std::endl;
    std::cout << "================================================================" << std::endl;

    // Parameters table
    std::cout << "\n--- Parameters ---" << std::endl;
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(11) << "multDepth"
              << std::setw(11) << "batchSize"
              << std::setw(5) << "d"
              << std::setw(5) << "r"
              << std::setw(16) << "bootstrapping"
              << std::setw(10) << "rotKeys" << std::endl;

    for (auto& r : results) {
        if (!r.valid) continue;
        bool isNaive = (r.name == "Naive");
        std::cout << std::left << std::setw(12) << r.name
                  << std::right << std::setw(11)
                  << (isNaive ? LinearRegression_Naive::MULT_DEPTH : 30)
                  << std::setw(11) << (isNaive ? 1 : SAMPLE_DIM * SAMPLE_DIM)
                  << std::setw(5) << (isNaive ? LinearRegression_Naive::D : FEATURE_DIM)
                  << std::setw(5) << (isNaive ? LinearRegression_Naive::INV_ITER : 18)
                  << std::setw(16) << (isNaive ? "No" : "Yes")
                  << std::setw(10) << (isNaive ? "No" : "Yes") << std::endl;
    }

    // Timing table
    std::cout << "\n--- Timing (seconds) ---" << std::endl;
    std::cout << std::left << std::setw(12) << "Step";
    for (auto& r : results) {
        if (r.valid) std::cout << std::right << std::setw(14) << r.name;
    }
    std::cout << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left << std::setw(12) << "Step 1";
    for (auto& r : results) {
        if (r.valid) std::cout << std::right << std::setw(14) << r.step1.count();
    }
    std::cout << std::endl;

    std::cout << std::left << std::setw(12) << "Step 2";
    for (auto& r : results) {
        if (r.valid) std::cout << std::right << std::setw(14) << r.step2.count();
    }
    std::cout << std::endl;

    std::cout << std::left << std::setw(12) << "Step 3";
    for (auto& r : results) {
        if (r.valid) std::cout << std::right << std::setw(14) << r.step3.count();
    }
    std::cout << std::endl;

    std::cout << std::left << std::setw(12) << "Total";
    for (auto& r : results) {
        if (r.valid) std::cout << std::right << std::setw(14) << r.total.count();
    }
    std::cout << std::endl;

    std::cout << "\n  Step 1: Precomputation -- X^T*X and X^T*y" << std::endl;
    std::cout << "    Naive: entry-level loops (n*d*(d+1) multiplications)" << std::endl;
    std::cout << "    AR24/NewCol: JKLS18 64x64 mult + rebatch 64->8 + Xty folding" << std::endl;
    std::cout << "  Step 2: Matrix Inversion -- (X^T*X)^{-1}" << std::endl;
    std::cout << "    Naive: I/trace(M) + " << LinearRegression_Naive::INV_ITER
              << " iterations (no bootstrapping)" << std::endl;
    std::cout << "    AR24: I/trace(M) + 18 iterations (AR24 mult, bootstrapping)" << std::endl;
    std::cout << "    NewCol: I/trace(M) + 18 iterations (NewCol mult, bootstrapping)" << std::endl;
    std::cout << "  Step 3: Weight Computation -- w = (X^T*X)^{-1} * X^T*y" << std::endl;
    std::cout << "    Naive: entry-level mat-vec multiply" << std::endl;
    std::cout << "    AR24/NewCol: packed mat-vec multiply" << std::endl;

    // Accuracy table
    std::cout << "\n--- Accuracy ---" << std::endl;
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(12) << "MSE" << std::endl;
    for (auto& r : results) {
        if (!r.valid) continue;
        std::cout << std::left << std::setw(12) << r.name
                  << std::right << std::setw(12) << std::setprecision(4)
                  << r.regResult.mse << std::endl;
    }

    // Serialized sizes table
    std::cout << "\n--- Serialized Sizes (Client<->Server Communication) ---" << std::endl;
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(10) << "CT(MB)"
              << std::setw(10) << "CT Count"
              << std::setw(14) << "EvalKeys(MB)"
              << std::setw(14) << "RelinKey(MB)"
              << std::setw(12) << "Total(MB)" << std::endl;

    for (auto& r : results) {
        if (!r.valid) continue;
        double totalMB = r.mem.ciphertextSizeMB * r.ctCount
                       + r.mem.rotationKeysSizeMB
                       + r.mem.relinKeySizeMB;
        std::cout << std::left << std::setw(12) << r.name
                  << std::right << std::setprecision(2)
                  << std::setw(10) << r.mem.ciphertextSizeMB
                  << std::setw(10) << r.ctCount
                  << std::setw(14) << r.mem.rotationKeysSizeMB
                  << std::setw(14) << r.mem.relinKeySizeMB
                  << std::setw(12) << totalMB << std::endl;
    }

    std::cout << "\n  CT Count = total ciphertexts (client->server)" << std::endl;
    std::cout << "    Naive: n_samples x d_features + n_samples = "
              << LinearRegression_Naive::N << " x " << LinearRegression_Naive::D
              << " + " << LinearRegression_Naive::N << " = "
              << (LinearRegression_Naive::N * LinearRegression_Naive::D + LinearRegression_Naive::N)
              << std::endl;
    std::cout << "    AR24/NewCol: X(1) + y(1) = 2" << std::endl;
    std::cout << "  EvalKeys = rotation keys + bootstrapping keys (Naive has none)" << std::endl;

    // Memory table
    std::cout << "\n--- Process Memory ---" << std::endl;
    std::cout << std::left << std::setw(12) << "Algorithm"
              << std::right << std::setw(14) << "Setup OH(GB)"
              << std::setw(16) << "Runtime OH(GB)"
              << std::setw(12) << "Peak(GB)" << std::endl;

    for (auto& r : results) {
        if (!r.valid) continue;
        std::cout << std::left << std::setw(12) << r.name
                  << std::right << std::setprecision(4)
                  << std::setw(14) << r.mem.setupOverheadGB()
                  << std::setw(16) << r.mem.computeOverheadGB()
                  << std::setw(12) << r.mem.peakMemoryGB << std::endl;
    }

    std::cout << "================================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    bool verbose = true;
    bool runNaiveFlag = false;
    bool runAR24Flag = false;
    bool runNewColFlag = false;
    bool runAll = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--naive") runNaiveFlag = true;
        else if (arg == "--ar24") runAR24Flag = true;
        else if (arg == "--newcol") runNewColFlag = true;
        else if (arg == "--all") runAll = true;
        else if (arg == "--benchmark") { verbose = false; }
    }

    // Default: run all if nothing specified
    if (!runNaiveFlag && !runAR24Flag && !runNewColFlag && !runAll) {
        runAll = true;
    }
    if (runAll) {
        runNaiveFlag = runAR24Flag = runNewColFlag = true;
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    Linear Regression - Naive vs Packed Comparison           #" << std::endl;
    std::cout << "#    Naive (entry-level) vs AR24 vs NewCol                    #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    std::string trainFile = std::string(DATA_DIR) + "/trainSet.csv";
    std::string testFile = std::string(DATA_DIR) + "/testSet.csv";

    double idleMemGB = MemoryMonitor::getMemoryUsageGB();
    std::cout << "\nIdle memory: " << std::fixed << std::setprecision(4)
              << idleMemGB << " GB" << std::endl;

    std::vector<AlgorithmResult> results;

    // Run Naive
    if (runNaiveFlag) {
        results.push_back(runNaive(trainFile, testFile, idleMemGB, verbose));
    }

    // Run AR24
    if (runAR24Flag) {
        results.push_back(runPacked<LinearRegression_AR24>(
            "AR24", trainFile, testFile, idleMemGB, verbose));
    }

    // Run NewCol
    if (runNewColFlag) {
        results.push_back(runPacked<LinearRegression_NewCol>(
            "NewCol", trainFile, testFile, idleMemGB, verbose));
    }

    // Print comparison table (only if multiple algorithms ran)
    int validCount = 0;
    for (auto& r : results) if (r.valid) validCount++;
    if (validCount > 0) {
        printComparisonTable(results);
    }

    std::cout << "\n" << std::string(64, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(64, '=') << std::endl;

    return 0;
}
