#include "fh_data_encoder.h"
#include "fh_he_trainer.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

int main() {
#ifdef DATASET_DIABETES
    std::string trainPath = std::string(DATA_DIR) + "/diabetes_train.csv";
    std::string testPath = std::string(DATA_DIR) + "/diabetes_test.csv";
    std::string datasetName = "Diabetes (8 features, 64 train)";
#else
    std::string trainPath = (FH_BATCH_SIZE == 64)
        ? std::string(DATA_DIR) + "/heart_train_64.csv"
        : std::string(DATA_DIR) + "/heart_train_128.csv";
    std::string testPath = std::string(DATA_DIR) + "/heart_test.csv";
    std::string datasetName = "Heart Disease (13 features, 128 train)";
#endif

    std::cout << "========================================" << std::endl;
    std::cout << " Logistic Regression (HE-friendly)" << std::endl;
    std::cout << " Plaintext Simulation — " << datasetName << std::endl;
    std::cout << "========================================" << std::endl;

    // Load datasets
    auto trainSet = FHDataEncoder::loadCSV(trainPath);
    auto testSet = FHDataEncoder::loadCSV(testPath);

    // Normalize
    FHDataEncoder::normalizeFeatures(trainSet);
    FHDataEncoder::normalizeWithParams(testSet, trainSet);

    // Add bias and pad
    FHDataEncoder::addBiasAndPad(trainSet);
    FHDataEncoder::addBiasAndPad(testSet);

    std::cout << "Train: " << trainSet.numSamples << " samples, "
              << FH_RAW_FEATURES << " features" << std::endl;
    std::cout << "Test: " << testSet.numSamples << " samples" << std::endl;

    const int NUM_BATCHES = 1;
    auto pre = FHHETrainer::precompute(trainSet, NUM_BATCHES, false);

#ifdef DATASET_DIABETES
    // Diabetes: needs more iterations for SFH to converge
    std::vector<int> iters = {1, 2, 4, 8, 16, 32, 64, 128, 256};
#else
    std::vector<int> iters = {1, 2, 4, 8, 16, 32, 64};
#endif

    // ============ Simplified Fixed Hessian ============
    std::cout << "\n=== Simplified Fixed Hessian ===" << std::endl;
    for (int nIter : iters) {
        auto result = FHHETrainer::trainSimplified(pre, nIter, false);
        auto inf = FHHETrainer::inference(result, testSet, false);
        std::cout << "iter=" << std::setw(3) << nIter
                  << " Accuracy: " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
    }

    // ============ Full Fixed Hessian ============
    std::cout << "\n=== Full Fixed Hessian ===" << std::endl;
    for (int nIter : iters) {
        auto result = FHHETrainer::trainFixed(pre, nIter, false);
        auto inf = FHHETrainer::inference(result, testSet, false);
        std::cout << "iter=" << std::setw(3) << nIter
                  << " Accuracy: " << std::fixed << std::setprecision(4)
                  << (double)inf.correct / inf.total << std::endl;
    }

    return 0;
}
