// main_encrypted.cpp
// Encrypted Linear Regression with AR24 vs NewCol comparison
// CKKS homomorphic encryption

#include "lr_newcol.h"
#include "lr_ar24.h"
#include "../../common/evaluation_metrics.h"
#include "encryption.h"
#include "rotation.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

// Result structure for algorithm comparison
struct LRExperimentResult {
    EvalMetrics::RegressionResult regResult;
    std::chrono::duration<double> totalTime{0};
    bool valid = false;
};

// Run Linear Regression with a given algorithm
template<typename LRAlgorithm>
LRExperimentResult runLinearRegression(const std::string& algorithmName,
                                        std::shared_ptr<Encryption> enc,
                                        CryptoContext<DCRTPoly> cc,
                                        KeyPair<DCRTPoly> keyPair,
                                        const std::vector<int>& rotIndices,
                                        int multDepth,
                                        const Ciphertext<DCRTPoly>& X,
                                        const Ciphertext<DCRTPoly>& y,
                                        const std::string& testFile,
                                        bool verbose) {
    using namespace std::chrono;
    LRExperimentResult expResult;

    EvalMetrics::printExperimentHeader("Linear Regression", algorithmName,
        SAMPLE_DIM, SAMPLE_DIM, FEATURE_DIM);
    std::cout << std::string(60, '-') << std::endl;

    LRAlgorithm lr(enc, cc, keyPair, rotIndices, multDepth);
    lr.setVerbose(verbose);

    // ========== Training ==========
    auto totalStart = high_resolution_clock::now();

    auto [step1_time, step2_time, step3_time, step4_time] = lr.trainWithTimings(X, y);

    auto totalEnd = high_resolution_clock::now();
    expResult.totalTime = totalEnd - totalStart;
    expResult.valid = true;

    // ========== Inference and MSE Calculation ==========
    std::cout << "\n--- Inference on Test Set ---" << std::endl;

    // Decrypt weights
    Plaintext ptx;
    cc->Decrypt(keyPair.secretKey, lr.getWeights(), &ptx);
    ptx->SetLength(FEATURE_DIM * FEATURE_DIM);
    std::vector<double> weights = ptx->GetRealPackedValue();

    // Print weights
    std::cout << "  Weights: ";
    for (int j = 0; j < std::min(FEATURE_DIM, 8); j++) {
        std::cout << std::setw(10) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << std::endl;

    // Load test data and compute predictions
    std::vector<double> test_features;
    std::vector<double> test_outcomes;
    CSVProcessor::processDataset(testFile, test_features, test_outcomes, FEATURE_DIM, SAMPLE_DIM);

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

    // Compute regression metrics
    expResult.regResult.compute(predictions, groundTruth);
    expResult.regResult.print(algorithmName);

    // Print timing summary
    EvalMetrics::TimingResult timing;
    timing.step1 = step1_time;
    timing.step2 = step2_time;
    timing.step3 = step3_time;
    timing.step4 = step4_time;
    timing.total = expResult.totalTime;
    timing.step1Name = "X^T X computation";
    timing.step2Name = "Matrix inversion";
    timing.step3Name = "X^T y computation";
    timing.step4Name = "Weight computation";
    timing.print(algorithmName);

    EvalMetrics::printExperimentFooter();

    return expResult;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    bool verbose = true;
    std::string algorithm = "both";  // "ar24", "newcol", "both"

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            verbose = false;
        } else if (arg == "--ar24") {
            algorithm = "ar24";
        } else if (arg == "--newcol") {
            algorithm = "newcol";
        }
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    Linear Regression - Encrypted Mode                       #" << std::endl;
    std::cout << "#    AR24 vs NewCol Matrix Inversion Comparison               #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Training samples: " << SAMPLE_DIM << std::endl;
    std::cout << "  Test samples:     " << SAMPLE_DIM << std::endl;
    std::cout << "  Features:         " << FEATURE_DIM << std::endl;
    std::cout << "  Matrix dimension: " << SAMPLE_DIM << "x" << SAMPLE_DIM << std::endl;

    // ========== Setup CKKS Encryption ==========
    int multDepth = 28;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(SAMPLE_DIM * SAMPLE_DIM);
    parameters.SetSecurityLevel(HEStd_128_classic);

    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();

    // Setup rotation keys
    std::vector<int> rotations;
    for (int i = 1; i < 1 << 16; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, FEATURE_DIM * FEATURE_DIM);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, FEATURE_DIM * FEATURE_DIM);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

    std::cout << "\n--- Encrypting Training Data ---" << std::endl;

    // Load and encrypt training data
    std::vector<double> features;
    std::vector<double> outcomes;
    CSVProcessor::processDataset(std::string(DATA_DIR) + "/trainSet.csv", features, outcomes,
                                 FEATURE_DIM, SAMPLE_DIM);

    auto X = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(features, 1, 0, nullptr, SAMPLE_DIM * SAMPLE_DIM));
    auto y = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(outcomes, 1, 0, nullptr, SAMPLE_DIM));

    std::cout << "  X encrypted: " << SAMPLE_DIM * SAMPLE_DIM << " slots" << std::endl;
    std::cout << "  y encrypted: " << SAMPLE_DIM << " slots" << std::endl;

    std::string testFile = std::string(DATA_DIR) + "/testSet.csv";

    // ========== Run Algorithms ==========
    LRExperimentResult ar24Result, newcolResult;

    if (algorithm == "ar24" || algorithm == "both") {
        ar24Result = runLinearRegression<LinearRegression_AR24>(
            "AR24", enc, cc, keyPair, rotations, multDepth,
            X, y, testFile, verbose);
    }

    if (algorithm == "newcol" || algorithm == "both") {
        newcolResult = runLinearRegression<LinearRegression_NewCol>(
            "NewCol", enc, cc, keyPair, rotations, multDepth,
            X, y, testFile, verbose);
    }

    // Print comparison summary if both algorithms were run
    if (ar24Result.valid && newcolResult.valid) {
        EvalMetrics::TimingResult ar24Timing, newcolTiming;
        ar24Timing.total = ar24Result.totalTime;
        newcolTiming.total = newcolResult.totalTime;

        EvalMetrics::printComparisonSummary(
            "Linear Regression",
            ar24Timing, newcolTiming,
            ar24Result.regResult.mse,
            newcolResult.regResult.mse,
            "MSE");
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
