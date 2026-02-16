// evaluation_metrics.h
// Unified evaluation metrics for all ML applications
#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

namespace EvalMetrics {

// ============================================================
// Classification Metrics (LDA, Fixed Hessian)
// ============================================================
struct ClassificationResult {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    int correct = 0, total = 0;

    double accuracy() const {
        return total > 0 ? 100.0 * correct / total : 0.0;
    }

    double precision() const {
        return (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0.0;
    }

    double recall() const {
        return (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0.0;
    }

    double f1Score() const {
        double p = precision();
        double r = recall();
        return (p + r) > 0 ? 2.0 * p * r / (p + r) : 0.0;
    }

    void print(const std::string& methodName, std::ostream& os = std::cout) const {
        os << "\n--- " << methodName << " Classification Results ---" << std::endl;
        os << std::fixed << std::setprecision(2);
        os << "  Correct:   " << correct << " / " << total << std::endl;
        os << "  Accuracy:  " << accuracy() << "%" << std::endl;
        os << "  Precision: " << precision() << "%" << std::endl;
        os << "  Recall:    " << recall() << "%" << std::endl;
        os << "  F1 Score:  " << f1Score() << "%" << std::endl;
        os << "  (TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn << ")" << std::endl;
    }
};

// Evaluate binary classification
// labels: {-1, +1} or {0, 1}
// predictions: same format as labels
template<typename Dataset>
ClassificationResult evaluateClassification(
    const std::vector<double>& weights,
    const Dataset& testSet,
    int numFeatures,
    double threshold = 0.0,
    bool useLinearDecision = true) {

    ClassificationResult result;
    result.total = testSet.numSamples;

    for (size_t i = 0; i < testSet.numSamples; i++) {
        double z = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            z += testSet.samples[i][j] * weights[j];
        }

        int pred_label;
        if (useLinearDecision) {
            // Linear decision: sign(z - threshold)
            pred_label = (z >= threshold) ? 1 : -1;
        } else {
            // Sigmoid approximation for logistic regression
            double pred_prob = 0.5 + z / 4.0;
            pred_label = (pred_prob >= 0.5) ? 1 : -1;
        }

        int true_label = testSet.labels[i];

        // Convert 0 to -1 if needed
        if (true_label == 0) true_label = -1;

        if (pred_label == true_label) result.correct++;

        if (pred_label == 1 && true_label == 1) result.tp++;
        else if (pred_label == 1 && true_label == -1) result.fp++;
        else if (pred_label == -1 && true_label == 1) result.fn++;
        else result.tn++;
    }

    return result;
}

// ============================================================
// Regression Metrics (Linear Regression)
// ============================================================
struct RegressionResult {
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double r2 = 0.0;
    int numSamples = 0;

    void compute(const std::vector<double>& predictions,
                 const std::vector<double>& groundTruth) {
        numSamples = predictions.size();
        if (numSamples == 0) return;

        double sumSqError = 0.0;
        double sumAbsError = 0.0;
        double sumY = 0.0;
        double sumYSq = 0.0;

        for (size_t i = 0; i < predictions.size(); i++) {
            double err = predictions[i] - groundTruth[i];
            sumSqError += err * err;
            sumAbsError += std::abs(err);
            sumY += groundTruth[i];
            sumYSq += groundTruth[i] * groundTruth[i];
        }

        mse = sumSqError / numSamples;
        rmse = std::sqrt(mse);
        mae = sumAbsError / numSamples;

        // R² = 1 - SS_res / SS_tot
        double meanY = sumY / numSamples;
        double ssTot = sumYSq - numSamples * meanY * meanY;
        r2 = (ssTot > 1e-10) ? 1.0 - sumSqError / ssTot : 0.0;
    }

    void print(const std::string& methodName, std::ostream& os = std::cout) const {
        os << "\n--- " << methodName << " Regression Results ---" << std::endl;
        os << std::fixed;
        os << "  Samples: " << numSamples << std::endl;
        os << "  MSE:     " << std::setprecision(6) << mse << std::endl;
        os << "  RMSE:    " << std::setprecision(6) << rmse << std::endl;
        os << "  MAE:     " << std::setprecision(6) << mae << std::endl;
        os << "  R^2:     " << std::setprecision(4) << r2 << std::endl;
    }
};

// ============================================================
// Timing Utilities
// ============================================================
struct TimingResult {
    std::chrono::duration<double> step1{0};  // Data preparation / X^T X
    std::chrono::duration<double> step2{0};  // Matrix inversion
    std::chrono::duration<double> step3{0};  // Weight computation
    std::chrono::duration<double> step4{0};  // Additional step (optional)
    std::chrono::duration<double> total{0};

    std::string step1Name = "Step 1";
    std::string step2Name = "Step 2";
    std::string step3Name = "Step 3";
    std::string step4Name = "Step 4";

    void print(const std::string& methodName, std::ostream& os = std::cout) const {
        os << "\n--- " << methodName << " Timing ---" << std::endl;
        os << std::fixed << std::setprecision(3);
        os << "  " << std::setw(20) << std::left << step1Name << ": " << step1.count() << " s" << std::endl;
        os << "  " << std::setw(20) << std::left << step2Name << ": " << step2.count() << " s" << std::endl;
        os << "  " << std::setw(20) << std::left << step3Name << ": " << step3.count() << " s" << std::endl;
        if (step4.count() > 0) {
            os << "  " << std::setw(20) << std::left << step4Name << ": " << step4.count() << " s" << std::endl;
        }
        os << "  " << std::string(30, '-') << std::endl;
        os << "  " << std::setw(20) << std::left << "TOTAL" << ": " << total.count() << " s" << std::endl;
    }
};

// ============================================================
// Unified Experiment Header/Footer
// ============================================================
inline void printExperimentHeader(const std::string& appName,
                                   const std::string& algorithmName,
                                   int trainSamples,
                                   int testSamples,
                                   int numFeatures,
                                   std::ostream& os = std::cout) {
    os << "\n" << std::string(60, '=') << std::endl;
    os << "  " << appName << " - " << algorithmName << std::endl;
    os << std::string(60, '=') << std::endl;
    os << "  Training samples: " << trainSamples << std::endl;
    os << "  Test samples:     " << testSamples << std::endl;
    os << "  Features:         " << numFeatures << std::endl;
    os << std::string(60, '-') << std::endl;
}

inline void printExperimentFooter(std::ostream& os = std::cout) {
    os << std::string(60, '=') << std::endl;
}

// ============================================================
// Comparison Summary (AR24 vs NewCol)
// ============================================================
inline void printComparisonSummary(
    const std::string& appName,
    const TimingResult& ar24Timing,
    const TimingResult& newcolTiming,
    double ar24Metric,   // Accuracy or MSE
    double newcolMetric,
    const std::string& metricName = "Accuracy",
    std::ostream& os = std::cout) {

    os << "\n" << std::string(60, '=') << std::endl;
    os << "  " << appName << " - Algorithm Comparison Summary" << std::endl;
    os << std::string(60, '=') << std::endl;
    os << std::fixed << std::setprecision(2);
    os << "  Algorithm     |  Time (s)  |  " << metricName << std::endl;
    os << "  " << std::string(50, '-') << std::endl;
    os << "  AR24          |  " << std::setw(8) << ar24Timing.total.count()
       << "  |  " << ar24Metric << (metricName == "Accuracy" ? "%" : "") << std::endl;
    os << "  NewCol        |  " << std::setw(8) << newcolTiming.total.count()
       << "  |  " << newcolMetric << (metricName == "Accuracy" ? "%" : "") << std::endl;
    os << std::string(60, '=') << std::endl;

    // Speedup calculation
    double speedup = newcolTiming.total.count() / ar24Timing.total.count();
    os << "  AR24 speedup over NewCol: " << std::setprecision(2) << speedup << "x" << std::endl;
    os << std::string(60, '=') << std::endl;
}

}  // namespace EvalMetrics
