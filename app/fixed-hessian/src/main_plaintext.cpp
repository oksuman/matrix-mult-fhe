#include "lr_data_encoder.h"
#include "lr_he_trainer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main() {
    std::string trainPath = std::string(DATA_DIR) + "/heart_train.csv";
    std::string testPath = std::string(DATA_DIR) + "/heart_test.csv";

    std::cout << "========================================" << std::endl;
    std::cout << " Logistic Regression (HE-friendly)" << std::endl;
    std::cout << " Plaintext Simulation — Heart Disease" << std::endl;
    std::cout << "========================================" << std::endl;

    // Step 1: Load data
    std::cout << "\n[1] Loading datasets..." << std::endl;
    auto trainSet = LRDataEncoder::loadCSV(trainPath);
    auto testSet = LRDataEncoder::loadCSV(testPath);

    LRDataEncoder::printDatasetInfo(trainSet, "Raw Train");
    LRDataEncoder::printDatasetInfo(testSet, "Raw Test");

    // Step 2: Min-max normalization to [0,1] (on 13 raw features)
    std::cout << "[2] Min-max normalization [0,1]..." << std::endl;
    LRDataEncoder::normalizeFeatures(trainSet);
    LRDataEncoder::normalizeWithParams(testSet, trainSet);

    // Step 3: Add bias column + padding (13 -> 16)
    std::cout << "[3] Adding bias column and padding to 16 features..." << std::endl;
    LRDataEncoder::addBiasAndPad(trainSet);
    LRDataEncoder::addBiasAndPad(testSet);

    LRDataEncoder::printDatasetInfo(trainSet, "Processed Train");
    LRDataEncoder::printDatasetInfo(testSet, "Processed Test");

    // 64 training samples = 1 batch of 64
    const int NUM_BATCHES = 1;

    // Step 4: Precompute (Xty, XtX, Hessian)
    std::cout << "\n[4] Precomputing batch statistics..." << std::endl;
    auto pre = LRHETrainer::precompute(trainSet, NUM_BATCHES, true);

    // Step 5: Simplified Fixed Hessian
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << " Simplified Fixed Hessian (Diagonal)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<int> simplifiedIters = {1};
    for (int nIter : simplifiedIters) {
        std::cout << "\n--- Simplified, iter=" << nIter << " ---" << std::endl;
        auto result = LRHETrainer::trainSimplified(pre, nIter, true);

        std::cout << "\nWeights: ";
        for (int j = 0; j < LR_RAW_FEATURES; j++) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << result.weights[j] << " ";
        }
        std::cout << "\nBias (w[13]): " << result.weights[LR_RAW_FEATURES] << std::endl;

        std::cout << "\nInference on test set:" << std::endl;
        LRHETrainer::inference(result, testSet, true);
    }

    // Step 6: Fixed Hessian (Full matrix)
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << " Fixed Hessian (Full 16x16 Matrix)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<int> fixedIters = {1};
    for (int nIter : fixedIters) {
        std::cout << "\n--- Fixed, iter=" << nIter << " ---" << std::endl;
        auto result = LRHETrainer::trainFixed(pre, nIter, true);

        std::cout << "\nWeights: ";
        for (int j = 0; j < LR_RAW_FEATURES; j++) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << result.weights[j] << " ";
        }
        std::cout << "\nBias (w[13]): " << result.weights[LR_RAW_FEATURES] << std::endl;

        std::cout << "\nInference on test set:" << std::endl;
        LRHETrainer::inference(result, testSet, true);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Done." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
