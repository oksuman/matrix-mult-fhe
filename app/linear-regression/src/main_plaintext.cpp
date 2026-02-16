// main_plaintext.cpp - Plaintext Linear Regression for debugging
#include "lr_plaintext_ops.h"
#include "csv_processor.h"
#include <iostream>
#include <fstream>
#include <chrono>

const int FEATURE_DIM = 8;
const int SAMPLE_DIM = 64;
const int TEST_SAMPLE_DIM = 256;

struct LRResult {
    std::vector<double> XtX;           // X^T * X (64x64)
    std::vector<double> XtX_rebatched; // Rebatched to 8x8
    std::vector<double> XtX_inv;       // (X^T * X)^{-1} (8x8)
    std::vector<double> Xty;           // X^T * y (8x1)
    std::vector<double> weights;       // Final weights (8x1)
    double mse;
};

LRResult runPlaintextLR(const std::string& trainFile, const std::string& testFile,
                        int inversionIterations, bool verbose) {
    LRResult result;

    std::cout << "\n========== Plaintext Linear Regression ==========" << std::endl;

    // Load training data
    std::vector<double> features, outcomes;
    CSVProcessor::processDataset(trainFile, features, outcomes, FEATURE_DIM, SAMPLE_DIM);

    std::cout << "Loaded " << SAMPLE_DIM << " samples, " << FEATURE_DIM << " features" << std::endl;

    // X is 64x64 (SAMPLE_DIM x SAMPLE_DIM), stored row-major
    // Only first FEATURE_DIM columns have data, rest are 0
    std::vector<double>& X = features;  // 64x64

    // y is 64x1 vector
    std::vector<double>& y = outcomes;

    if (verbose) {
        LRPlaintextOps::printMatrix("X (first 5 rows)", X, SAMPLE_DIM, SAMPLE_DIM, 5);
        LRPlaintextOps::printVector("y (first 8)", y, 8);
    }

    // Step 1: X^T * X
    std::cout << "\n[Step 1] Computing X^T * X..." << std::endl;
    auto Xt = LRPlaintextOps::transpose(X, SAMPLE_DIM);
    result.XtX = LRPlaintextOps::matMult(Xt, X, SAMPLE_DIM);

    if (verbose) {
        LRPlaintextOps::printMatrix("X^T * X (64x64, top-left 8x8)", result.XtX, SAMPLE_DIM, SAMPLE_DIM, 8);
    }

    // Rebatch: Extract 8x8 from 64x64
    std::cout << "[Step 1b] Rebatching X^T*X from 64x64 to 8x8..." << std::endl;
    result.XtX_rebatched = LRPlaintextOps::rebatch(result.XtX, SAMPLE_DIM, FEATURE_DIM);

    if (verbose) {
        LRPlaintextOps::printMatrix("X^T * X (rebatched 8x8)", result.XtX_rebatched, FEATURE_DIM, FEATURE_DIM, 8);
        double tr = LRPlaintextOps::trace(result.XtX_rebatched, FEATURE_DIM);
        std::cout << "trace(X^T * X) = " << tr << std::endl;
    }

    // Step 2: (X^T * X)^{-1}
    std::cout << "\n[Step 2] Computing (X^T * X)^{-1}..." << std::endl;
    result.XtX_inv = LRPlaintextOps::invertMatrix(result.XtX_rebatched, FEATURE_DIM,
                                                   inversionIterations, verbose);

    if (verbose) {
        LRPlaintextOps::printMatrix("(X^T * X)^{-1}", result.XtX_inv, FEATURE_DIM, FEATURE_DIM, 8);

        // Verify: (X^T X)^{-1} * (X^T X) should be I
        auto check = LRPlaintextOps::matMult(result.XtX_inv, result.XtX_rebatched, FEATURE_DIM);
        std::cout << "Verification: (X^T X)^{-1} * (X^T X) diagonal: ";
        for (int i = 0; i < FEATURE_DIM; i++) {
            std::cout << std::setprecision(4) << check[i * FEATURE_DIM + i] << " ";
        }
        std::cout << std::endl;
    }

    // Step 3: X^T * y
    std::cout << "\n[Step 3] Computing X^T * y..." << std::endl;
    // y is already a 64x1 vector
    std::vector<double>& y_vec = y;

    // X^T * y using Xt (64x64) * y_vec (64x1) -> 64x1, then take first 8
    auto Xty_full = LRPlaintextOps::matVecMult(Xt, y_vec, SAMPLE_DIM, SAMPLE_DIM);
    result.Xty = std::vector<double>(Xty_full.begin(), Xty_full.begin() + FEATURE_DIM);

    if (verbose) {
        LRPlaintextOps::printVector("X^T * y", result.Xty, FEATURE_DIM);
    }

    // Step 4: weights = (X^T X)^{-1} * X^T y
    std::cout << "\n[Step 4] Computing weights = (X^T X)^{-1} * X^T y..." << std::endl;
    result.weights = LRPlaintextOps::matVecMult(result.XtX_inv, result.Xty, FEATURE_DIM, FEATURE_DIM);

    LRPlaintextOps::printVector("Weights", result.weights, FEATURE_DIM);

    std::cout << "\n[Step 4b] Simulating encrypted computation..." << std::endl;
    auto simulated = LRPlaintextOps::simulateEncryptedMatVecMult(result.XtX_inv, result.Xty, FEATURE_DIM, true);

    // Load test data and compute MSE
    std::cout << "\n[Inference] Computing MSE on test set (" << TEST_SAMPLE_DIM << " samples)..." << std::endl;
    std::vector<double> test_features, test_outcomes;
    CSVProcessor::processDataset(testFile, test_features, test_outcomes, FEATURE_DIM, TEST_SAMPLE_DIM);

    double mse = 0.0;
    int count = 0;
    for (int i = 0; i < TEST_SAMPLE_DIM; i++) {
        double pred = 0.0;
        for (int j = 0; j < FEATURE_DIM; j++) {
            pred += result.weights[j] * test_features[i * TEST_SAMPLE_DIM + j];
        }
        double actual = test_outcomes[i];
        mse += (pred - actual) * (pred - actual);
        count++;
    }
    result.mse = mse / count;

    std::cout << "MSE: " << result.mse << std::endl;

    return result;
}

int main() {
    std::string dataDir = std::string(DATA_DIR);
    std::string trainFile = dataDir + "/trainSet.csv";
    std::string testFile = dataDir + "/testSet.csv";

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    Linear Regression - Plaintext Mode                       #" << std::endl;
    std::cout << "#    For debugging encrypted implementation                   #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    auto result = runPlaintextLR(trainFile, testFile, 18, true);

    // Save results
    std::ofstream outFile("plaintext_lr_results.txt");
    outFile << "=== Plaintext Linear Regression Results ===" << std::endl;
    outFile << "\nWeights:" << std::endl;
    for (int i = 0; i < FEATURE_DIM; i++) {
        outFile << "  w[" << i << "] = " << std::setprecision(8) << result.weights[i] << std::endl;
    }
    outFile << "\nMSE: " << result.mse << std::endl;

    outFile << "\nX^T * X (8x8):" << std::endl;
    for (int i = 0; i < FEATURE_DIM; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            outFile << std::setw(14) << std::setprecision(6) << result.XtX_rebatched[i * FEATURE_DIM + j];
        }
        outFile << std::endl;
    }

    outFile << "\n(X^T * X)^{-1} (8x8):" << std::endl;
    for (int i = 0; i < FEATURE_DIM; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            outFile << std::setw(14) << std::setprecision(6) << result.XtX_inv[i * FEATURE_DIM + j];
        }
        outFile << std::endl;
    }

    outFile << "\nX^T * y:" << std::endl;
    for (int i = 0; i < FEATURE_DIM; i++) {
        outFile << "  " << std::setprecision(8) << result.Xty[i] << std::endl;
    }

    outFile.close();
    std::cout << "\nResults saved to plaintext_lr_results.txt" << std::endl;

    return 0;
}
