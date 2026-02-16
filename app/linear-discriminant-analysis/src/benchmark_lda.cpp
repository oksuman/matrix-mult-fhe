// benchmark_lda.cpp
// Benchmark comparison of NewCol vs AR24 matrix inversion for LDA
// Measures: S_W computation, S_W^{-1} computation, w computation
// Reports: Accuracy, F1 Score, and detailed timing
//
// Compile with -DBENCHMARK_VERBOSE=1 for debug output, or 0 for pure benchmark

#ifndef BENCHMARK_VERBOSE
#define BENCHMARK_VERBOSE 0  // Default: no debug output for accurate timing
#endif

#include "lda_data_encoder.h"
#include "lda_ar24.h"
#include "lda_newcol.h"
#include "encryption.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

// ============ Configuration ============
const int NUM_SAMPLES = 64;           // Training samples
const int INVERSION_ITERATIONS = 25;  // Inversion iterations
const bool USE_BOOTSTRAPPING = true;
const int NUM_TRIALS = 1;             // Number of trials for averaging

// ============ Benchmark Result Structure ============
struct BenchmarkResult {
    std::string algorithm;
    int num_trials;

    // Timing (seconds) - averaged over trials
    double time_sw;           // S_W computation
    double time_sw_inv;       // S_W^{-1} computation
    double time_w;            // w = S_W^{-1} * (mu_1 - mu_0) computation
    double time_total;        // Total training time

    // Metrics (from last trial - should be same across trials)
    double accuracy;
    double precision;
    double recall;
    double f1_score;

    // Confusion matrix
    int tp, tn, fp, fn;
};

// ============ Utility Functions ============
std::string getDataDir() {
    std::vector<std::string> paths = {
        std::string(DATA_DIR),
        "../data",
        "app/linear-discriminant-analysis/data",
        "data"
    };
    for (const auto& path : paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::string(DATA_DIR);
}

std::vector<int> generateRotationIndices(int maxDim) {
    std::vector<int> rotations;
    int batchSize = maxDim * maxDim;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    return rotations;
}

std::vector<Ciphertext<DCRTPoly>> encryptClassData(
    const LDADataset& dataset,
    const EncodedData& encoded,
    std::shared_ptr<Encryption> enc) {

    std::vector<Ciphertext<DCRTPoly>> classDataEncrypted;
    for (size_t c = 0; c < dataset.numClasses; c++) {
        auto ct = enc->encryptInput(encoded.classSamples[c]);
        classDataEncrypted.push_back(ct);
    }
    return classDataEncrypted;
}

// ============ Inference with Metrics ============
BenchmarkResult performInferenceWithMetrics(
    const LDAEncryptedResult& trainResult,
    const LDADataset& testSet,
    const LDATimingResult& timings,
    double time_w,
    const std::string& algorithm) {

    BenchmarkResult result;
    result.algorithm = algorithm;
    result.time_sw = timings.swComputation.count();
    result.time_sw_inv = timings.inversionTime.count();
    result.time_w = time_w;
    result.time_total = timings.totalTime.count();

    size_t f = testSet.numFeatures;
    size_t f_tilde = testSet.paddedFeatures;

    // Compute Fisher direction
    std::vector<double> mu_diff(f, 0.0);
    for (size_t i = 0; i < f; i++) {
        mu_diff[i] = trainResult.classMeans[1][i] - trainResult.classMeans[0][i];
    }

    std::vector<double> w(f, 0.0);
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < f; j++) {
            w[i] += trainResult.Sw_inv_decrypted[i * f_tilde + j] * mu_diff[j];
        }
    }

    // Project class means
    double proj_mu0 = 0.0, proj_mu1 = 0.0;
    for (size_t i = 0; i < f; i++) {
        proj_mu0 += w[i] * trainResult.classMeans[0][i];
        proj_mu1 += w[i] * trainResult.classMeans[1][i];
    }

    // Threshold
    double n0 = trainResult.classCounts[0];
    double n1 = trainResult.classCounts[1];
    double threshold = (n1 * proj_mu0 + n0 * proj_mu1) / (n0 + n1);

    // Classify and compute confusion matrix
    result.tp = result.tn = result.fp = result.fn = 0;

    for (size_t s = 0; s < testSet.numSamples; s++) {
        double projection = 0.0;
        for (size_t i = 0; i < f; i++) {
            projection += w[i] * testSet.samples[s][i];
        }

        int predicted = (proj_mu1 > proj_mu0) ?
            (projection > threshold ? 1 : 0) :
            (projection < threshold ? 1 : 0);

        int actual = testSet.labels[s];

        if (predicted == 1 && actual == 1) result.tp++;
        else if (predicted == 0 && actual == 0) result.tn++;
        else if (predicted == 1 && actual == 0) result.fp++;
        else if (predicted == 0 && actual == 1) result.fn++;
    }

    // Compute metrics
    int total = result.tp + result.tn + result.fp + result.fn;
    result.accuracy = 100.0 * (result.tp + result.tn) / total;

    result.precision = (result.tp + result.fp > 0) ?
        (double)result.tp / (result.tp + result.fp) : 0.0;
    result.recall = (result.tp + result.fn > 0) ?
        (double)result.tp / (result.tp + result.fn) : 0.0;
    result.f1_score = (result.precision + result.recall > 0) ?
        2.0 * result.precision * result.recall / (result.precision + result.recall) : 0.0;

    return result;
}

// ============ Save Results to File ============
void saveBenchmarkResults(const std::string& filename,
                          const BenchmarkResult& newcol,
                          const BenchmarkResult& ar24,
                          int numSamples,
                          int inversionIters) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return;
    }

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    file << "================================================================\n";
    file << "  LDA Benchmark Results\n";
    file << "  Generated: " << std::ctime(&time);
    file << "================================================================\n\n";

    file << "--- Configuration ---\n";
    file << "Training samples: " << numSamples << "\n";
    file << "Inversion iterations: " << inversionIters << "\n";
    file << "Bootstrapping: " << (USE_BOOTSTRAPPING ? "enabled" : "disabled") << "\n";
    file << "Number of trials: " << newcol.num_trials << "\n\n";

    file << std::fixed << std::setprecision(4);

    file << "================================================================\n";
    file << "  TIMING COMPARISON (seconds)\n";
    file << "================================================================\n\n";

    file << std::setw(25) << "Step"
         << std::setw(15) << "NewCol"
         << std::setw(15) << "AR24"
         << std::setw(15) << "Diff" << "\n";
    file << std::string(70, '-') << "\n";

    file << std::setw(25) << "S_W computation"
         << std::setw(15) << newcol.time_sw
         << std::setw(15) << ar24.time_sw
         << std::setw(15) << (ar24.time_sw - newcol.time_sw) << "\n";

    file << std::setw(25) << "S_W^{-1} computation"
         << std::setw(15) << newcol.time_sw_inv
         << std::setw(15) << ar24.time_sw_inv
         << std::setw(15) << (ar24.time_sw_inv - newcol.time_sw_inv) << "\n";

    file << std::setw(25) << "w computation"
         << std::setw(15) << newcol.time_w
         << std::setw(15) << ar24.time_w
         << std::setw(15) << (ar24.time_w - newcol.time_w) << "\n";

    file << std::string(70, '-') << "\n";
    file << std::setw(25) << "TOTAL"
         << std::setw(15) << newcol.time_total
         << std::setw(15) << ar24.time_total
         << std::setw(15) << (ar24.time_total - newcol.time_total) << "\n\n";

    file << "================================================================\n";
    file << "  ACCURACY & F1 SCORE\n";
    file << "================================================================\n\n";

    file << std::setw(25) << "Metric"
         << std::setw(15) << "NewCol"
         << std::setw(15) << "AR24" << "\n";
    file << std::string(55, '-') << "\n";

    file << std::setw(25) << "Accuracy (%)"
         << std::setw(15) << newcol.accuracy
         << std::setw(15) << ar24.accuracy << "\n";

    file << std::setw(25) << "Precision"
         << std::setw(15) << newcol.precision
         << std::setw(15) << ar24.precision << "\n";

    file << std::setw(25) << "Recall"
         << std::setw(15) << newcol.recall
         << std::setw(15) << ar24.recall << "\n";

    file << std::setw(25) << "F1 Score"
         << std::setw(15) << newcol.f1_score
         << std::setw(15) << ar24.f1_score << "\n\n";

    file << "================================================================\n";
    file << "  CONFUSION MATRICES\n";
    file << "================================================================\n\n";

    file << "NewCol:\n";
    file << "              Predicted\n";
    file << "              0      1\n";
    file << "Actual  0   " << std::setw(4) << newcol.tn << "   " << std::setw(4) << newcol.fp << "\n";
    file << "        1   " << std::setw(4) << newcol.fn << "   " << std::setw(4) << newcol.tp << "\n\n";

    file << "AR24:\n";
    file << "              Predicted\n";
    file << "              0      1\n";
    file << "Actual  0   " << std::setw(4) << ar24.tn << "   " << std::setw(4) << ar24.fp << "\n";
    file << "        1   " << std::setw(4) << ar24.fn << "   " << std::setw(4) << ar24.tp << "\n\n";

    file << "================================================================\n";
    file << "  END OF REPORT\n";
    file << "================================================================\n";

    file.close();
    std::cout << "\nResults saved to: " << filename << std::endl;
}

// ============ Run Single Algorithm with Multiple Trials ============
template<typename LDAAlgorithm>
BenchmarkResult runBenchmark(
    const std::string& algorithmName,
    std::shared_ptr<Encryption> enc,
    CryptoContext<DCRTPoly> cc,
    KeyPair<DCRTPoly> keyPair,
    const std::vector<int>& rotIndices,
    const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
    const LDADataset& trainSet,
    const LDADataset& testSet,
    int multDepth) {

    std::cout << "Running " << algorithmName << " (" << NUM_TRIALS << " trial"
              << (NUM_TRIALS > 1 ? "s" : "") << ")..." << std::flush;

    // Accumulate timing over trials
    double total_time_sw = 0, total_time_inv = 0, total_time_w = 0, total_time_total = 0;
    BenchmarkResult finalResult;
    finalResult.algorithm = algorithmName;
    finalResult.num_trials = NUM_TRIALS;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
#if BENCHMARK_VERBOSE
        std::cout << "\n  Trial " << (trial + 1) << "/" << NUM_TRIALS << ":" << std::endl;
#endif

        LDAAlgorithm lda(enc, cc, keyPair, rotIndices, multDepth, USE_BOOTSTRAPPING);

        LDATimingResult timings;
        auto result = lda.trainWithTimings(classDataEncrypted, trainSet,
                                            INVERSION_ITERATIONS, timings,
                                            BENCHMARK_VERBOSE, false);

        // Measure w computation time separately
        auto wStart = std::chrono::high_resolution_clock::now();

        size_t f = testSet.numFeatures;
        size_t f_tilde = testSet.paddedFeatures;
        std::vector<double> mu_diff(f, 0.0);
        for (size_t i = 0; i < f; i++) {
            mu_diff[i] = result.classMeans[1][i] - result.classMeans[0][i];
        }
        std::vector<double> w(f, 0.0);
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                w[i] += result.Sw_inv_decrypted[i * f_tilde + j] * mu_diff[j];
            }
        }

        auto wEnd = std::chrono::high_resolution_clock::now();
        double time_w = std::chrono::duration<double>(wEnd - wStart).count();

        // Get metrics (will be same for all trials)
        finalResult = performInferenceWithMetrics(result, testSet, timings, time_w, algorithmName);

        // Accumulate timing
        total_time_sw += timings.swComputation.count();
        total_time_inv += timings.inversionTime.count();
        total_time_w += time_w;
        total_time_total += timings.totalTime.count();

#if BENCHMARK_VERBOSE
        std::cout << "    Time: " << std::fixed << std::setprecision(1)
                  << timings.totalTime.count() << "s" << std::endl;
#endif
    }

    // Compute averages
    finalResult.num_trials = NUM_TRIALS;
    finalResult.time_sw = total_time_sw / NUM_TRIALS;
    finalResult.time_sw_inv = total_time_inv / NUM_TRIALS;
    finalResult.time_w = total_time_w / NUM_TRIALS;
    finalResult.time_total = total_time_total / NUM_TRIALS;

    std::cout << " Done. (avg: " << std::fixed << std::setprecision(1)
              << finalResult.time_total << "s)" << std::endl;

    return finalResult;
}

// ============ Main ============
int main() {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    std::cout << "\n";
    std::cout << "###############################################################\n";
    std::cout << "#                                                             #\n";
    std::cout << "#    LDA Benchmark: NewCol vs AR24                            #\n";
    std::cout << "#    Heart Disease Dataset                                    #\n";
    std::cout << "#                                                             #\n";
    std::cout << "###############################################################\n\n";

    std::string dataDir = getDataDir();
    std::cout << "Data directory: " << dataDir << std::endl;

    // Load dataset
    std::cout << "\n--- Loading Dataset ---" << std::endl;
    LDADataset trainSet, testSet;
    LDADataEncoder::loadOrCreateSplit(
        dataDir + "/Heart_disease_cleveland.csv",
        dataDir + "/heart_disease_train.csv",
        dataDir + "/heart_disease_test.csv",
        trainSet, testSet, 0.8, 42);

    // Limit training samples
    if (NUM_SAMPLES > 0) {
        std::cout << "Limiting training samples to " << NUM_SAMPLES << std::endl;
        LDADataEncoder::limitSamples(trainSet, NUM_SAMPLES);
    }

    std::cout << "Training: " << trainSet.numSamples << " samples" << std::endl;
    std::cout << "Test: " << testSet.numSamples << " samples" << std::endl;
    std::cout << "Features: " << trainSet.numFeatures << " (padded: " << trainSet.paddedFeatures << ")" << std::endl;

    // Normalize
    LDADataEncoder::normalizeFeatures(trainSet);
    LDADataEncoder::normalizeWithParams(testSet, trainSet);

    // Encode data
    std::cout << "\n--- Encoding Data ---" << std::endl;
    int largeDim = std::max(trainSet.paddedSamples, trainSet.paddedFeatures);
    EncodedData encoded = LDADataEncoder::encode(trainSet, largeDim);

    // Setup encryption
    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;
    int maxDim = std::max(trainSet.paddedSamples, trainSet.paddedFeatures);
    int multDepth = 28;
    uint32_t scalingModSize = 59;
    uint32_t firstModSize = 60;

    auto rotIndices = generateRotationIndices(maxDim);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(maxDim * maxDim);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    std::cout << "Generating rotation keys..." << std::flush;
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);
    std::cout << " Done." << std::endl;

    std::cout << "Setting up bootstrapping..." << std::flush;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
    std::cout << " Done." << std::endl;

    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);

    // Encrypt data
    std::cout << "\n--- Encrypting Training Data ---" << std::endl;
    auto classDataEncrypted = encryptClassData(trainSet, encoded, enc);
    std::cout << "Encrypted " << classDataEncrypted.size() << " class datasets" << std::endl;

    // Run benchmarks
    auto newcolResult = runBenchmark<LDA_NewCol>(
        "NewCol", enc, cc, keyPair, rotIndices,
        classDataEncrypted, trainSet, testSet, multDepth);

    auto ar24Result = runBenchmark<LDA_AR24>(
        "AR24", enc, cc, keyPair, rotIndices,
        classDataEncrypted, trainSet, testSet, multDepth);

    // Save results
    std::string outputFile = "benchmark_results.txt";
    saveBenchmarkResults(outputFile, newcolResult, ar24Result, NUM_SAMPLES, INVERSION_ITERATIONS);

    // Print summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nNewCol: " << newcolResult.accuracy << "% accuracy, "
              << "F1=" << std::setprecision(4) << newcolResult.f1_score
              << ", Total=" << std::setprecision(1) << newcolResult.time_total << "s" << std::endl;
    std::cout << "AR24:   " << std::setprecision(2) << ar24Result.accuracy << "% accuracy, "
              << "F1=" << std::setprecision(4) << ar24Result.f1_score
              << ", Total=" << std::setprecision(1) << ar24Result.time_total << "s" << std::endl;

    double speedup = newcolResult.time_total / ar24Result.time_total;
    std::cout << "\nSpeedup (NewCol/AR24): " << std::setprecision(2) << speedup << "x" << std::endl;

    std::cout << "\nResults saved to: " << outputFile << std::endl;

    return 0;
}
