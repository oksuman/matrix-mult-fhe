#include "lr_data_encoder.h"
#include "lr_he_trainer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main() {
    std::string trainPath = (LR_BATCH_SIZE == 64)
        ? std::string(DATA_DIR) + "/heart_train.csv"
        : std::string(DATA_DIR) + "/heart_combined_128.csv";
    std::string testPath64 = std::string(DATA_DIR) + "/heart_test.csv";
    std::string testPath128 = std::string(DATA_DIR) + "/heart_test_128.csv";

    std::cout << "========================================" << std::endl;
    std::cout << " Logistic Regression (HE-friendly)" << std::endl;
    std::cout << " Plaintext Simulation — Heart Disease" << std::endl;
    std::cout << "========================================" << std::endl;

    // Load datasets
    auto trainSet = LRDataEncoder::loadCSV(trainPath);
    auto testSet64 = LRDataEncoder::loadCSV(testPath64);
    auto testSet128 = LRDataEncoder::loadCSV(testPath128);

    // Normalize
    LRDataEncoder::normalizeFeatures(trainSet);
    LRDataEncoder::normalizeWithParams(testSet64, trainSet);
    LRDataEncoder::normalizeWithParams(testSet128, trainSet);

    // Add bias and pad
    LRDataEncoder::addBiasAndPad(trainSet);
    LRDataEncoder::addBiasAndPad(testSet64);
    LRDataEncoder::addBiasAndPad(testSet128);

    const int NUM_BATCHES = 1;
    auto pre = LRHETrainer::precompute(trainSet, NUM_BATCHES, false);

    std::vector<int> iters = {1, 2, 4, 8, 16, 32, 64};

    // ============ 64 test samples ============
    std::cout << "\n=== Simplified 64test ===" << std::endl;
    for (int nIter : iters) {
        auto result = LRHETrainer::trainSimplified(pre, nIter, false);
        auto inf = LRHETrainer::inference(result, testSet64, false);
        std::cout << "Test Accuracy:  " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
    }

    std::cout << "\n=== Full Fixed 64test ===" << std::endl;
    for (int nIter : iters) {
        auto result = LRHETrainer::trainFixed(pre, nIter, false);
        auto inf = LRHETrainer::inference(result, testSet64, false);
        std::cout << "Test Accuracy:  " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
    }

    // ============ 128 test samples ============
    std::cout << "\n=== Simplified 128test ===" << std::endl;
    for (int nIter : iters) {
        auto result = LRHETrainer::trainSimplified(pre, nIter, (nIter <= 2));
        auto inf = LRHETrainer::inference(result, testSet128, false);
        std::cout << "Test Accuracy:  " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
        // Print weights for comparison with encrypted
        if (nIter == 1) {
            std::cout << "  Weights: ";
            for (int j = 0; j < 14; j++) {
                std::cout << std::setw(8) << std::setprecision(4) << result.weights[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n=== Full Fixed 128test ===" << std::endl;
    for (int nIter : iters) {
        auto result = LRHETrainer::trainFixed(pre, nIter, false);
        auto inf = LRHETrainer::inference(result, testSet128, false);
        std::cout << "Test Accuracy:  " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
    }

    return 0;
}
