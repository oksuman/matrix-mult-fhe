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
#include <chrono>
#include <filesystem>
#include <cmath>
#include <set>

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
std::vector<int> generateRotationIndices(int maxDim) {
    std::set<int> indices;

    // Binary decomposition rotations
    for (int i = 1; i <= maxDim * maxDim; i *= 2) {
        indices.insert(i);
        indices.insert(-i);
    }

    // Matrix transpose rotations
    for (int d = 1; d <= maxDim; d *= 2) {
        for (int k = 1; k < d; k++) {
            indices.insert((d - 1) * k);
            indices.insert(-(d - 1) * k);
        }
    }

    // JKLS18 / AR24 specific rotations
    for (int i = 0; i < maxDim; i++) {
        indices.insert(i);
        indices.insert(-i);
        indices.insert(i * maxDim);
        indices.insert(-i * maxDim);
    }

    // NewCol specific rotations
    for (int i = 1; i <= maxDim; i++) {
        for (int j = 0; j < maxDim; j++) {
            indices.insert(i + j * maxDim);
            indices.insert(-(i + j * maxDim));
        }
    }

    // Folding sum rotations
    for (int i = 1; i < maxDim; i *= 2) {
        indices.insert(i * HD_PADDED_FEATURE);
        indices.insert(-i * HD_PADDED_FEATURE);
    }

    return std::vector<int>(indices.begin(), indices.end());
}

// Encrypt dataset for training
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

// Run encrypted LDA experiment
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
                     bool verbose) {

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  LDA with " << algorithmName << " Matrix Inversion" << std::endl;
    std::cout << "  Bootstrapping: " << (useBootstrapping ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    LDAAlgorithm lda(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);

    LDATimingResult timings;
    auto result = lda.trainWithTimings(classDataEncrypted, trainSet, inversionIterations, timings, verbose);

    // Perform inference on test set
    std::cout << "\n--- Inference on Test Set ---" << std::endl;
    double accuracy = performInference(result, testSet, verbose);

    std::cout << "\nTest Accuracy: " << std::setprecision(2) << std::fixed
              << accuracy << "%" << std::endl;

    std::cout << "\n--- Timing Summary ---" << std::endl;
    std::cout << "Mean computation:   " << std::setprecision(3)
              << timings.meanComputation.count() << " s" << std::endl;
    std::cout << "S_W computation:    " << timings.swComputation.count() << " s" << std::endl;
    std::cout << "S_B computation:    " << timings.sbComputation.count() << " s" << std::endl;
    std::cout << "Matrix inversion:   " << timings.inversionTime.count() << " s" << std::endl;
    std::cout << "Total training:     " << timings.totalTime.count() << " s" << std::endl;
}

int main(int argc, char* argv[]) {
    bool verbose = true;
    bool useBootstrapping = true;
    std::string algorithm = "both";  // "ar24", "newcol", or "both"

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-q" || arg == "--quiet") {
            verbose = false;
        } else if (arg == "--no-bootstrap") {
            useBootstrapping = false;
        } else if (arg == "--ar24") {
            algorithm = "ar24";
        } else if (arg == "--newcol") {
            algorithm = "newcol";
        }
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

    LDADataEncoder::printDatasetInfo(trainSet, "Training Set");
    LDADataEncoder::printDatasetInfo(testSet, "Test Set");

    // Normalize
    LDADataEncoder::normalizeFeatures(trainSet);
    LDADataEncoder::normalizeWithParams(testSet, trainSet);

    // Encode (CKKS-style packing)
    std::cout << "\n--- Encoding Data ---" << std::endl;
    auto encodedTrain = LDADataEncoder::encode(trainSet);

    std::cout << "Features: " << trainSet.numFeatures << " (padded: " << trainSet.paddedFeatures << ")" << std::endl;
    std::cout << "Samples: " << trainSet.numSamples << " (padded: " << trainSet.paddedSamples << ")" << std::endl;

    // ========== Setup CKKS Encryption ==========
    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;

    int maxDim = HD_MATRIX_DIM;  // 256 for HD dataset
    int multDepth = 45;  // High depth for matrix operations + inversion
    uint32_t scalingModSize = 50;
    uint32_t firstModSize = 60;

    if (!useBootstrapping) {
        multDepth = 30;  // Lower depth when not bootstrapping
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

    // Generate rotation keys
    std::cout << "Generating rotation keys..." << std::endl;
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);

    if (useBootstrapping) {
        std::cout << "Setting up bootstrapping..." << std::endl;
        std::vector<uint32_t> levelBudget = {4, 5};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
        cc->EvalBootstrapKeyGen(keyPair.secretKey, HD_PADDED_FEATURE * HD_PADDED_FEATURE);
    }

    // ========== Encrypt Training Data ==========
    std::cout << "\n--- Encrypting Training Data ---" << std::endl;
    auto classDataEncrypted = encryptClassData(trainSet, encodedTrain, enc);
    std::cout << "Encrypted " << classDataEncrypted.size() << " class datasets" << std::endl;

    // ========== Run Encrypted LDA ==========
    int inversionIterations = 20;  // Schulz iterations for 16×16 matrix

    if (algorithm == "ar24" || algorithm == "both") {
        runEncryptedLDA<LDA_AR24>(
            "AR24",
            enc, cc, keyPair, rotIndices,
            classDataEncrypted, trainSet, testSet,
            multDepth, inversionIterations, useBootstrapping, verbose);
    }

    if (algorithm == "newcol" || algorithm == "both") {
        runEncryptedLDA<LDA_NewCol>(
            "NewCol",
            enc, cc, keyPair, rotIndices,
            classDataEncrypted, trainSet, testSet,
            multDepth, inversionIterations, useBootstrapping, verbose);
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
