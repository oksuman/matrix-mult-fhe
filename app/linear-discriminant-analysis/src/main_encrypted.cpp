// main_encrypted.cpp
// Encrypted LDA (Linear Discriminant Analysis) for Heart Disease dataset
// Compares AR24 and NewCol matrix inversion algorithms

#include "lda_data_encoder.h"
#include "lda_ar24.h"
#include "lda_newcol.h"
#include "lda_inference.h"
#include "encryption.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cmath>

using namespace lbcrypto;

// Get the data directory path
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

// Generate rotation indices needed for encrypted operations
// Only power-of-2 rotations are needed since RotationComposer uses binary decomposition
std::vector<int> generateRotationIndices(int maxDim) {
    std::vector<int> rotations;
    int batchSize = maxDim * maxDim;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    return rotations;
}

// Encrypt dataset for training (client-side)
// Client sends: per-class encrypted samples + sample counts (plaintext)
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

// Perform plaintext inference using encrypted training results
double performInference(const LDAEncryptedResult& trainResult,
                       const LDADataset& testSet,
                       bool verbose = false) {
    // Compute Fisher direction: w = S_W^{-1} * (mu_1 - mu_0)
    size_t f = testSet.numFeatures;
    size_t f_tilde = testSet.paddedFeatures;

    std::vector<double> mu_diff(f, 0.0);
    for (size_t i = 0; i < f; i++) {
        mu_diff[i] = trainResult.classMeans[1][i] - trainResult.classMeans[0][i];
    }

    // w = S_W^{-1} * (mu_1 - mu_0)
    std::vector<double> w(f, 0.0);
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < f; j++) {
            w[i] += trainResult.Sw_inv_decrypted[i * f_tilde + j] * mu_diff[j];
        }
    }

    // Project class means onto w
    double proj_mu0 = 0.0, proj_mu1 = 0.0;
    for (size_t i = 0; i < f; i++) {
        proj_mu0 += w[i] * trainResult.classMeans[0][i];
        proj_mu1 += w[i] * trainResult.classMeans[1][i];
    }

    // Weighted threshold (sklearn style)
    double n0 = trainResult.classCounts[0];
    double n1 = trainResult.classCounts[1];
    double threshold = (n1 * proj_mu0 + n0 * proj_mu1) / (n0 + n1);

    if (verbose) {
        std::cout << "\nFisher direction (first 5): ";
        for (size_t i = 0; i < std::min((size_t)5, f); i++) {
            std::cout << std::setprecision(4) << w[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Projected mu_0: " << proj_mu0 << std::endl;
        std::cout << "Projected mu_1: " << proj_mu1 << std::endl;
        std::cout << "Threshold: " << threshold << std::endl;
    }

    // Classify test samples
    int correct = 0;
    int total = 0;

    for (size_t s = 0; s < testSet.numSamples; s++) {
        double projection = 0.0;
        for (size_t i = 0; i < f; i++) {
            projection += w[i] * testSet.samples[s][i];
        }

        // Decision: if projection > threshold, predict class 1
        int predicted = (proj_mu1 > proj_mu0) ?
            (projection > threshold ? 1 : 0) :
            (projection < threshold ? 1 : 0);

        if (predicted == testSet.labels[s]) {
            correct++;
        }
        total++;
    }

    double accuracy = 100.0 * correct / total;
    return accuracy;
}

void saveEncryptedResults(const std::string& filename,
                          const LDAEncryptedResult& result,
                          const LDADataset& dataset,
                          double accuracy,
                          const LDATimingResult& timings) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    size_t f = dataset.numFeatures;
    size_t f_tilde = dataset.paddedFeatures;

    file << std::fixed << std::setprecision(6);

    file << "=== Global Mean (len=" << f << ") ===" << std::endl;
    for (size_t i = 0; i < f; i++) {
        file << std::setw(10) << result.globalMean[i] << " ";
    }
    file << std::endl << std::endl;

    for (size_t c = 0; c < result.classMeans.size(); c++) {
        file << "=== Class " << c << " Mean (len=" << f << ") ===" << std::endl;
        for (size_t i = 0; i < f; i++) {
            file << std::setw(10) << result.classMeans[c][i] << " ";
        }
        file << std::endl << std::endl;
    }

    file << "=== S_B (" << f << "x" << f << ") ===" << std::endl;
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < f; j++) {
            file << std::setw(10) << result.Sb_decrypted[i * f_tilde + j] << " ";
        }
        file << std::endl;
    }
    file << std::endl;

    // Intermediate results: X_bar_c and S_c per class
    int largeDim = HD_MATRIX_DIM;  // 256
    for (size_t c = 0; c < result.X_bar_c_decrypted.size(); c++) {
        size_t s_c = result.classCounts[c];

        file << "=== X_bar_c (class " << c << ", first 10 rows, " << f << " cols) ===" << std::endl;
        for (size_t row = 0; row < 10 && row < s_c; row++) {
            for (size_t col = 0; col < f; col++) {
                file << std::setw(10) << result.X_bar_c_decrypted[c][row * largeDim + col] << " ";
            }
            file << std::endl;
        }
        file << std::endl;

        file << "=== S_c (class " << c << " scatter, " << f << "x" << f << ", 256x256 top-left) ===" << std::endl;
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                file << std::setw(10) << result.S_c_decrypted[c][i * largeDim + j] << " ";
            }
            file << std::endl;
        }
        file << std::endl;
    }

    file << "=== S_W (" << f << "x" << f << ") ===" << std::endl;
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < f; j++) {
            file << std::setw(10) << result.Sw_decrypted[i * f_tilde + j] << " ";
        }
        file << std::endl;
    }
    file << std::endl;

    file << "=== S_W^{-1} (" << f << "x" << f << ") ===" << std::endl;
    for (size_t i = 0; i < f; i++) {
        for (size_t j = 0; j < f; j++) {
            file << std::setw(10) << result.Sw_inv_decrypted[i * f_tilde + j] << " ";
        }
        file << std::endl;
    }
    file << std::endl;

    file << "=== Accuracy ===" << std::endl;
    file << accuracy << "%" << std::endl << std::endl;

    file << "=== Timing ===" << std::endl;
    file << "Mean computation: " << timings.meanComputation.count() << " s" << std::endl;
    file << "S_B computation: " << timings.sbComputation.count() << " s" << std::endl;
    file << "S_W computation: " << timings.swComputation.count() << " s" << std::endl;
    file << "Matrix inversion: " << timings.inversionTime.count() << " s" << std::endl;
    file << "Total: " << timings.totalTime.count() << " s" << std::endl;

    file.close();
    std::cout << "Results saved to: " << filename << std::endl;
}

template<typename LDAAlgorithm>
void runEncryptedLDA(const std::string& algorithmName,
                     std::shared_ptr<Encryption> enc,
                     CryptoContext<DCRTPoly> cc,
                     KeyPair<DCRTPoly> keyPair,
                     const std::vector<int>& rotIndices,
                     const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
                     const LDADataset& trainSet,
                     const LDADataset& testSet,
                     int multDepth,
                     int inversionIterations,
                     bool useBootstrapping,
                     bool verbose,
                     bool sbOnly = false,
                     const std::string& outputFile = "") {

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  LDA with " << algorithmName << " Matrix Inversion" << std::endl;
    std::cout << "  Bootstrapping: " << (useBootstrapping ? "ENABLED" : "DISABLED") << std::endl;
    if (sbOnly) std::cout << "  Mode: S_B ONLY (quick test)" << std::endl;
    std::cout << std::string(60, '=') << std::flush;

    LDAAlgorithm lda(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);

    LDATimingResult timings;
    auto result = lda.trainWithTimings(classDataEncrypted, trainSet, inversionIterations, timings, verbose, sbOnly);

    // Skip inference and full output in sbOnly mode
    if (sbOnly) {
        std::cout << "\n--- S_B Only Mode Complete ---" << std::endl;
        return;
    }

    std::cout << "\n--- Inference on Test Set ---" << std::endl << std::flush;
    double accuracy = performInference(result, testSet, verbose);

    std::cout << "\nTest Accuracy: " << std::setprecision(2) << std::fixed
              << accuracy << "%" << std::endl;

    std::cout << "\n--- Timing Summary ---" << std::endl;
    std::cout << "Mean computation:   " << std::setprecision(3)
              << timings.meanComputation.count() << " s" << std::endl;
    std::cout << "S_B computation:    " << timings.sbComputation.count() << " s" << std::endl;
    std::cout << "S_W computation:    " << timings.swComputation.count() << " s" << std::endl;
    std::cout << "Matrix inversion:   " << timings.inversionTime.count() << " s" << std::endl;
    std::cout << "Total training:     " << timings.totalTime.count() << std::endl << std::flush;

    if (!outputFile.empty()) {
        saveEncryptedResults(outputFile, result, trainSet, accuracy, timings);
    }
}

int main(int argc, char* argv[]) {
    bool debugMode = true;
    bool useBootstrapping = true;
    std::string algorithm = "both";
    int maxTrainSamples = 64;  // Fixed to 64 samples

    bool sbOnly = false;  // Stop after S_B computation (for quick testing)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            debugMode = false;
        } else if (arg == "--no-bootstrap") {
            useBootstrapping = false;
        } else if (arg == "--ar24") {
            algorithm = "ar24";
        } else if (arg == "--newcol") {
            algorithm = "newcol";
        } else if (arg == "--sb-only") {
            sbOnly = true;
        } else if (arg == "--train-samples" && i + 1 < argc) {
            maxTrainSamples = std::stoi(argv[++i]);
        }
    }

    bool verbose = debugMode;
    std::string outputFilePrefix = debugMode ? "encrypted_" : "";

    if (sbOnly) {
        std::cout << "\n*** S_B ONLY MODE: Will stop after S_B computation ***\n" << std::endl;
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    LDA (Linear Discriminant Analysis) - Encrypted Mode      #" << std::endl;
    std::cout << "#    Heart Disease Dataset (HD)                               #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    std::string dataDir = getDataDir();
    std::cout << "\nData directory: " << dataDir << std::endl;

    // ========== Load Dataset ==========
    std::cout << "\n--- Loading Heart Disease Dataset ---" << std::endl;

    LDADataset trainSet, testSet;
    LDADataEncoder::loadOrCreateSplit(
        dataDir + "/Heart_disease_cleveland.csv",
        dataDir + "/heart_disease_train.csv",
        dataDir + "/heart_disease_test.csv",
        trainSet, testSet, 0.8, 42);

    // Limit training samples if specified
    if (maxTrainSamples > 0) {
        std::cout << "Limiting training samples to " << maxTrainSamples << std::endl;
        LDADataEncoder::limitSamples(trainSet, maxTrainSamples);
    }

    LDADataEncoder::printDatasetInfo(trainSet, "Training Set");
    LDADataEncoder::printDatasetInfo(testSet, "Test Set");

    // Normalize
    LDADataEncoder::normalizeFeatures(trainSet);
    LDADataEncoder::normalizeWithParams(testSet, trainSet);

    // Encode as matrices for JKLS18
    std::cout << "\n--- Encoding Data ---" << std::endl;
    int largeDim = std::max(trainSet.paddedSamples, trainSet.paddedFeatures);
    auto encodedTrain = LDADataEncoder::encode(trainSet, largeDim);

    std::cout << "Features: " << trainSet.numFeatures << " (padded: " << trainSet.paddedFeatures << ")" << std::endl;
    std::cout << "Samples: " << trainSet.numSamples << std::endl;
    std::cout << "Encoding: " << largeDim << "x" << largeDim << " = " << largeDim * largeDim << " slots" << std::endl;

    // ========== Setup CKKS Encryption ==========
    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;

    int maxDim = largeDim;  // 256 for HD dataset
    int multDepth;  // High depth for matrix operations + inversion
    uint32_t scalingModSize;
    uint32_t firstModSize;

    if (!useBootstrapping) { // without bootstrapping
        multDepth = 30; 
        scalingModSize = 50;
        firstModSize = 50;
    }else { // with bootstrapping
        multDepth = 29;
        scalingModSize = 59;
        firstModSize = 60;
    }

    std::cout << "Max dimension: " << maxDim << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Bootstrapping: " << (useBootstrapping ? "enabled" : "disabled") << std::endl;

    // Generate rotation indices
    auto rotIndices = generateRotationIndices(maxDim);
    std::cout << "Rotation indices: " << rotIndices.size() << std::endl;

    // Create CKKS parameters
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
    if (useBootstrapping) {
        cc->Enable(FHE);
    }

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // Create encryption object for data encryption
    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);

    std::cout << "Generating rotation keys..." << std::flush;
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);
    std::cout << " Done." << std::endl;

    if (useBootstrapping) {
        std::cout << "Setting up bootstrapping..." << std::flush;
        std::vector<uint32_t> levelBudget = {4, 5};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
        std::cout << " Setup done. Generating keys..." << std::flush;
        cc->EvalBootstrapKeyGen(keyPair.secretKey, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
        std::cout << " Done." << std::endl;
    }

    std::cout << "\n--- Encrypting Training Data ---" << std::flush;
    auto classDataEncrypted = encryptClassData(trainSet, encodedTrain, enc);
    std::cout << " Encrypted " << classDataEncrypted.size() << " class datasets" << std::endl;

    // ========== Run Encrypted LDA ==========
    int inversionIterations = 25;  // iterations for 16×16 matrix

    if (algorithm == "ar24" || algorithm == "both") {
        std::string outFile = outputFilePrefix.empty() ? "" : outputFilePrefix + "ar24_results.txt";
        runEncryptedLDA<LDA_AR24>(
            "AR24",
            enc, cc, keyPair, rotIndices,
            classDataEncrypted, trainSet, testSet,
            multDepth, inversionIterations, useBootstrapping, verbose, sbOnly, outFile);
    }

    if (algorithm == "newcol" || algorithm == "both") {
        std::string outFile = outputFilePrefix.empty() ? "" : outputFilePrefix + "newcol_results.txt";
        runEncryptedLDA<LDA_NewCol>(
            "NewCol",
            enc, cc, keyPair, rotIndices,
            classDataEncrypted, trainSet, testSet,
            multDepth, inversionIterations, useBootstrapping, verbose, sbOnly, outFile);
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
