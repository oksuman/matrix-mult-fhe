#include "lda_data_encoder.h"
#include "lda_trainer.h"
#include "lda_inference.h"
#include <iostream>
#include <chrono>
#include <filesystem>

// Get the data directory path
std::string getDataDir() {
    std::vector<std::string> paths = {
        std::string(DATA_DIR),                          // CMake defined path
        "../data",                                       // From build/app/linear-discriminant-analysis
        "app/linear-discriminant-analysis/data",        // From project root
        "data"                                           // From app/linear-discriminant-analysis
    };

    for (const auto& path : paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::string(DATA_DIR);
}

void runLDA(const std::string& datasetName,
            const std::string& rawDataFile,
            const std::string& trainFile,
            const std::string& testFile,
            int inversionIterations,
            bool verbose = true,
            const std::string& outputFile = "") {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  LDA on " << datasetName << " Dataset" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Load or create train/test split
    LDADataset trainSet, testSet;
    std::cout << "\n--- Data Loading ---" << std::endl;
    LDADataEncoder::loadOrCreateSplit(rawDataFile, trainFile, testFile,
                                      trainSet, testSet, 0.8, 42);

    LDADataEncoder::printDatasetInfo(trainSet, "Training Set");
    LDADataEncoder::printDatasetInfo(testSet, "Test Set");

    // Normalize features (train set determines min/max)
    LDADataEncoder::normalizeFeatures(trainSet);
    LDADataEncoder::normalizeWithParams(testSet, trainSet);

    // Encode data
    std::cout << "--- Encoding Data (CKKS-style packing) ---" << std::endl;
    auto encodedTrain = LDADataEncoder::encode(trainSet);

    std::cout << "All samples vector length: " << encodedTrain.allSamples.size()
              << " (" << encodedTrain.paddedSamples << " x " << encodedTrain.paddedFeatures << ")" << std::endl;

    for (size_t c = 0; c < trainSet.numClasses; c++) {
        std::cout << "Class " << c << " vector length: " << encodedTrain.classSamples[c].size()
                  << " (" << encodedTrain.paddedSamplesPerClass[c] << " x " << encodedTrain.paddedFeatures << ")" << std::endl;
    }

    // Train LDA
    std::cout << "\n--- Training LDA ---" << std::endl;
    auto trainStart = std::chrono::high_resolution_clock::now();

    auto trainResult = LDATrainer::train(encodedTrain, trainSet, inversionIterations, verbose, outputFile);

    auto trainEnd = std::chrono::high_resolution_clock::now();
    auto trainDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);

    std::cout << "Training time: " << trainDuration.count() << " ms" << std::endl;
    std::cout << "Matrix dimension for inversion: " << trainResult.matrixDim << " x " << trainResult.matrixDim
              << " (padded: " << trainResult.paddedDim << " x " << trainResult.paddedDim << ")" << std::endl;

    // Inference on test set
    std::cout << "\n--- Inference on Test Set ---" << std::endl;
    auto inferResult = LDAInference::infer(trainResult, testSet, verbose);

    // Print results
    LDAInference::printConfusionMatrix(inferResult, testSet);
}

int main(int argc, char* argv[]) {
    bool verbose = true;
    bool saveResults = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-q" || arg == "--quiet") {
            verbose = false;
        } else if (arg == "--save") {
            saveResults = true;
        }
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    LDA (Linear Discriminant Analysis) - Plaintext Mode     #" << std::endl;
    std::cout << "#    CKKS-friendly structure for FHE compatibility           #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    std::string dataDir = getDataDir();
    std::cout << "\nData directory: " << dataDir << std::endl;

    runLDA("PID (Diabetes)",
           dataDir + "/diabetes.csv",
           dataDir + "/diabetes_train.csv",
           dataDir + "/diabetes_test.csv",
           25,
           verbose,
           saveResults ? "plaintext_pid_results.txt" : "");

    runLDA("HD (Heart Disease)",
           dataDir + "/Heart_disease_cleveland.csv",
           dataDir + "/heart_disease_train.csv",
           dataDir + "/heart_disease_test.csv",
           25,
           verbose,
           saveResults ? "plaintext_hd_results.txt" : "");

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
