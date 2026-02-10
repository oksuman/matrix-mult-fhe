#pragma once

#include "lda_data_encoder.h"
#include "lda_trainer.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// For binary LDA: project onto Fisher's discriminant direction
// w = S_W^{-1} * (mu_1 - mu_0)
// Prediction: if w^T * x > threshold then class 1, else class 0
// Threshold is typically w^T * (mu_0 + mu_1) / 2

struct InferenceResult {
    std::vector<int> predictions;
    int correctCount;
    int totalCount;
    double accuracy;
    double precision;
    double recall;
    double f1;
};

class LDAInference {
public:
    // Compute Fisher's discriminant direction for binary LDA
    // w = S_W^{-1} * (mu_1 - mu_0)
    static std::vector<double> computeFisherDirection(const LDATrainResult& trainResult) {
        size_t d = trainResult.matrixDim;

        // Compute diff = mu_1 - mu_0
        std::vector<double> diff(d);
        for (size_t i = 0; i < d; i++) {
            diff[i] = trainResult.classMeans[1][i] - trainResult.classMeans[0][i];
        }

        // Directly compute w = S_W^{-1} * (mu_1 - mu_0)
        // This is the correct Fisher discriminant direction
        std::vector<double> w(d, 0.0);
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                w[i] += trainResult.Sw_inv[i * d + j] * diff[j];
            }
        }

        return w;
    }

    // Compute projection threshold with prior probability weighting
    // This matches sklearn's LDA behavior
    static double computeThreshold(const std::vector<double>& w,
                                   const LDATrainResult& trainResult,
                                   size_t n0, size_t n1) {
        size_t d = trainResult.matrixDim;

        // Project class means onto w
        double proj_mu0 = 0.0, proj_mu1 = 0.0;
        for (size_t i = 0; i < d; i++) {
            proj_mu0 += w[i] * trainResult.classMeans[0][i];
            proj_mu1 += w[i] * trainResult.classMeans[1][i];
        }

        // Weighted threshold by class prior (matches sklearn)
        // Use opposite weights: majority class weight on minority mean
        // This incorporates log-prior into the decision boundary
        // threshold = (n1 * proj_mu0 + n0 * proj_mu1) / (n0 + n1)
        double threshold = (n1 * proj_mu0 + n0 * proj_mu1) / (n0 + n1);

        return threshold;
    }

    // Perform inference on test set
    static InferenceResult infer(const LDATrainResult& trainResult,
                                 const LDADataset& testSet,
                                 bool verbose = false) {
        InferenceResult result;

        size_t d = trainResult.matrixDim;
        size_t numTests = testSet.numSamples;

        // Compute Fisher direction and threshold
        auto w = computeFisherDirection(trainResult);
        double threshold = computeThreshold(w, trainResult,
                                            trainResult.classCounts[0],
                                            trainResult.classCounts[1]);

        // Determine direction: if w^T * mu_1 > w^T * mu_0, then class 1 is above threshold
        double proj_mu0 = 0.0, proj_mu1 = 0.0;
        for (size_t i = 0; i < d; i++) {
            proj_mu0 += w[i] * trainResult.classMeans[0][i];
            proj_mu1 += w[i] * trainResult.classMeans[1][i];
        }

        bool class1Above = (proj_mu1 > proj_mu0);

        if (verbose) {
            std::cout << "\n========== LDA Inference ==========" << std::endl;
            std::cout << "Fisher direction w:" << std::endl;
            for (size_t i = 0; i < d; i++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed << w[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Threshold: " << threshold << std::endl;
            std::cout << "Projection of mu_0: " << proj_mu0 << std::endl;
            std::cout << "Projection of mu_1: " << proj_mu1 << std::endl;
            std::cout << "Class 1 is " << (class1Above ? "above" : "below") << " threshold" << std::endl;
            std::cout << std::endl;
        }

        result.predictions.resize(numTests);
        result.correctCount = 0;

        for (size_t i = 0; i < numTests; i++) {
            // Project test sample onto w
            double projection = 0.0;
            for (size_t j = 0; j < d; j++) {
                projection += w[j] * testSet.samples[i][j];
            }

            // Predict class
            int predicted;
            if (class1Above) {
                predicted = (projection > threshold) ? 1 : 0;
            } else {
                predicted = (projection < threshold) ? 1 : 0;
            }

            result.predictions[i] = predicted;

            if (predicted == testSet.labels[i]) {
                result.correctCount++;
            }

            if (verbose && i < 10) {
                std::cout << "Sample " << i << ": proj=" << std::setw(8) << std::setprecision(4) << projection
                          << ", predicted=" << predicted
                          << ", actual=" << testSet.labels[i]
                          << (predicted == testSet.labels[i] ? " OK" : " WRONG")
                          << std::endl;
            }
        }

        result.totalCount = numTests;
        result.accuracy = static_cast<double>(result.correctCount) / numTests;

        // Compute precision, recall, F1
        int tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < numTests; i++) {
            int actual = testSet.labels[i];
            int predicted = result.predictions[i];
            if (actual == 1 && predicted == 1) tp++;
            else if (actual == 0 && predicted == 1) fp++;
            else if (actual == 1 && predicted == 0) fn++;
        }
        result.precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        result.recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        result.f1 = (result.precision + result.recall > 0) ?
                    2 * result.precision * result.recall / (result.precision + result.recall) : 0.0;

        if (verbose) {
            if (numTests > 10) {
                std::cout << "..." << std::endl;
            }
            std::cout << "\n========== Results ==========" << std::endl;
            std::cout << "Correct: " << result.correctCount << " / " << result.totalCount << std::endl;
            std::cout << "Accuracy: " << std::setprecision(2) << std::fixed << (result.accuracy * 100) << "%" << std::endl;
        }

        return result;
    }

    // Print confusion matrix
    static void printConfusionMatrix(const InferenceResult& result,
                                     const LDADataset& testSet) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < testSet.numSamples; i++) {
            int actual = testSet.labels[i];
            int predicted = result.predictions[i];

            if (actual == 1 && predicted == 1) tp++;
            else if (actual == 0 && predicted == 0) tn++;
            else if (actual == 0 && predicted == 1) fp++;
            else if (actual == 1 && predicted == 0) fn++;
        }

        std::cout << "\n=== Confusion Matrix ===" << std::endl;
        std::cout << "              Predicted" << std::endl;
        std::cout << "              0     1" << std::endl;
        std::cout << "Actual  0   " << std::setw(4) << tn << "  " << std::setw(4) << fp << std::endl;
        std::cout << "        1   " << std::setw(4) << fn << "  " << std::setw(4) << tp << std::endl;
        std::cout << std::endl;

        double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;

        std::cout << "Precision: " << std::setprecision(2) << std::fixed << (precision * 100) << "%" << std::endl;
        std::cout << "Recall: " << std::setprecision(2) << std::fixed << (recall * 100) << "%" << std::endl;
        std::cout << "F1 Score: " << std::setprecision(2) << std::fixed << (f1 * 100) << "%" << std::endl;
    }
};
