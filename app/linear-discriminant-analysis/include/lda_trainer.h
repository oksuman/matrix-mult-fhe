#pragma once

#include "lda_data_encoder.h"
#include "lda_plaintext_ops.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>

struct LDATrainResult {
    std::vector<double> Sw_inv_Sb;      // S_W^{-1} * S_B matrix (d x d)
    std::vector<double> Sw_inv;         // S_W^{-1} matrix (for direct Fisher computation)
    std::vector<std::vector<double>> classMeans;  // Per-class mean vectors
    std::vector<double> globalMean;     // Global mean vector
    std::vector<size_t> classCounts;    // Number of samples per class (for threshold)

    size_t matrixDim;   // Actual feature dimension
    size_t paddedDim;   // Padded dimension (power of 2)

    // Intermediate results for debugging
    std::vector<std::vector<double>> X_bar_c;   // Centered data per class
    std::vector<std::vector<double>> S_c;       // Scatter matrix per class

    // Fisher discriminant direction and projected means
    std::vector<double> eigenvector;     // w = S_W^{-1} * (mu_1 - mu_0)
    double projectedMean0;               // w^T * mu_0
    double projectedMean1;               // w^T * mu_1

    // Store Sw and Sb for file saving
    std::vector<double> Sw;
    std::vector<double> Sb;
};

class LDATrainer {
private:
    static std::vector<double> invertMatrix(const std::vector<double>& A,
                                            int d,
                                            int iterations = 20) {
        std::vector<double> I(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            I[i * d + i] = 1.0;
        }

        // Estimate trace for initial scaling
        double trace = 0.0;
        for (int i = 0; i < d; i++) {
            trace += A[i * d + i];
        }

        if (std::abs(trace) < 1e-10) {
            throw std::runtime_error("Matrix appears to be singular (trace ~ 0)");
        }

        // Y_0 = (1/trace(A)) * I
        std::vector<double> Y = PlaintextOps::multScalar(I, 1.0 / trace);

        // A_bar = I - A * Y_0
        auto AY = PlaintextOps::matMult(A, Y, d);
        std::vector<double> A_bar = PlaintextOps::sub(I, AY);

        // Iterate: Y = Y * (I + A_bar), A_bar = A_bar * A_bar
        for (int iter = 0; iter < iterations - 1; iter++) {
            auto I_plus_Abar = PlaintextOps::add(I, A_bar);
            Y = PlaintextOps::matMult(Y, I_plus_Abar, d);
            A_bar = PlaintextOps::matMult(A_bar, A_bar, d);
        }

        // Final iteration
        auto I_plus_Abar = PlaintextOps::add(I, A_bar);
        Y = PlaintextOps::matMult(Y, I_plus_Abar, d);

        return Y;
    }

public:
    static LDATrainResult train(const EncodedData& encoded,
                                const LDADataset& dataset,
                                int inversionIterations = 20,
                                bool verbose = false,
                                const std::string& outputFile = "") {
        LDATrainResult result;

        size_t f = dataset.numFeatures;
        size_t f_tilde = dataset.paddedFeatures;
        size_t s_tilde = dataset.paddedSamples;
        size_t numClasses = dataset.numClasses;

        result.matrixDim = f;
        result.paddedDim = f_tilde;
        result.classCounts = dataset.samplesPerClass;  // Store for threshold calculation

        if (verbose) {
            std::cout << "\n========== LDA Training (Plaintext CKKS-style) ==========" << std::endl;
            std::cout << "Features: " << f << " (padded: " << f_tilde << ")" << std::endl;
            std::cout << "Total samples: " << dataset.numSamples << " (padded: " << s_tilde << ")" << std::endl;
        }

        // ========== Step 1: Compute Global Mean ==========
        // Using folding sum: log(s̃) rotations and additions
        if (verbose) std::cout << "\n[Step 1] Computing global mean..." << std::endl;

        auto globalMeanReplicated = PlaintextOps::computeMean(
            encoded.allSamples,
            dataset.numSamples,
            s_tilde,
            f_tilde
        );
        auto globalMean = PlaintextOps::extractFirstRow(globalMeanReplicated, f_tilde);
        result.globalMean = std::vector<double>(globalMean.begin(), globalMean.begin() + f);

        if (verbose) {
            PlaintextOps::printVector(result.globalMean, f, "Global Mean");
        }

        // ========== Step 2: Compute Class Means ==========
        if (verbose) std::cout << "[Step 2] Computing class means..." << std::endl;

        result.classMeans.resize(numClasses);
        std::vector<std::vector<double>> classMeanReplicated(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            classMeanReplicated[c] = PlaintextOps::computeMean(
                encoded.classSamples[c],
                s_c,
                s_tilde_c,
                f_tilde
            );

            auto classMean = PlaintextOps::extractFirstRow(classMeanReplicated[c], f_tilde);
            result.classMeans[c] = std::vector<double>(classMean.begin(), classMean.begin() + f);

            if (verbose) {
                PlaintextOps::printVector(result.classMeans[c], f, "Class " + std::to_string(c) + " Mean");
            }
        }

        // ========== Step 3: Compute S_B (Between-class scatter) ==========
        // S_B = sum_c s_c * (mu_c - mu)(mu_c - mu)^T
        if (verbose) std::cout << "[Step 3] Computing S_B (between-class scatter)..." << std::endl;

        std::vector<double> Sb(f_tilde * f_tilde, 0.0);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];

            // diff = mu_c - mu
            auto classMean = PlaintextOps::extractFirstRow(classMeanReplicated[c], f_tilde);
            auto diff = PlaintextOps::sub(classMean, globalMean);

            if (verbose) {
                PlaintextOps::printVector(diff, f, "(mu_" + std::to_string(c) + " - mu)");
            }

            // outer = diff * diff^T
            auto outer = PlaintextOps::outerProduct(diff, diff, f, f_tilde);

            // Scale by class size and accumulate
            auto scaled = PlaintextOps::multScalar(outer, static_cast<double>(s_c));

            if (verbose) {
                std::cout << "  Scaled by n_" << c << " = " << s_c << std::endl;
            }

            PlaintextOps::addInPlace(Sb, scaled);
        }

        if (verbose) {
            PlaintextOps::printMatrix(Sb, f, f, "S_B", f_tilde);
        }

        // ========== Step 4: Compute S_W (Within-class scatter) ==========
        // S_W = sum_c sum_{x in class c} (x - mu_c)(x - mu_c)^T
        //     = sum_c (X_c - mu_c)^T * (X_c - mu_c)
        if (verbose) std::cout << "[Step 4] Computing S_W (within-class scatter)..." << std::endl;

        std::vector<double> Sw(f_tilde * f_tilde, 0.0);
        result.X_bar_c.resize(numClasses);
        result.S_c.resize(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            if (verbose) {
                std::cout << "  Class " << c << " (n=" << s_c << "):" << std::endl;
                std::cout << "    Computing X_bar = X - mu..." << std::endl;
            }

            // Replicate class mean to match encoded data shape
            auto meanReplicated = PlaintextOps::replicateRow(
                PlaintextOps::extractFirstRow(classMeanReplicated[c], f_tilde),
                s_tilde_c,
                f_tilde
            );

            // X_bar_c = X_c - mu_c (broadcast subtraction)
            auto X_bar_c = PlaintextOps::sub(encoded.classSamples[c], meanReplicated);
            result.X_bar_c[c] = X_bar_c;  // Store for debugging

            if (verbose) {
                // Print first few rows of X_bar_c (centered data)
                std::cout << "    X_bar_c (first 5 rows, first " << f << " cols):" << std::endl;
                for (size_t row = 0; row < 5 && row < s_c; row++) {
                    std::cout << "      Row " << row << ": ";
                    for (size_t col = 0; col < f; col++) {
                        std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                                  << X_bar_c[row * f_tilde + col] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                // Print X_bar_c^T (first f rows, first 5 cols) - simulated transpose view
                std::cout << "    X_bar_c^T (first " << f << " rows, first 5 cols):" << std::endl;
                for (size_t row = 0; row < f; row++) {
                    std::cout << "      Row " << row << ": ";
                    for (size_t col = 0; col < 5 && col < s_c; col++) {
                        // X_bar_c^T[row][col] = X_bar_c[col][row]
                        std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                                  << X_bar_c[col * f_tilde + row] << " ";
                    }
                    std::cout << "..." << std::endl;
                }
                std::cout << std::endl;
            }

            // S_c = X_bar_c^T * X_bar_c
            auto S_c = PlaintextOps::computeXtX(X_bar_c, s_c, f, s_tilde_c, f_tilde);
            result.S_c[c] = S_c;  // Store for debugging

            if (verbose) {
                PlaintextOps::printMatrix(S_c, f, f, "    S_c (class " + std::to_string(c) + " scatter)", f_tilde);
            }

            // Accumulate
            PlaintextOps::addInPlace(Sw, S_c);

            if (verbose) {
                std::cout << "    Accumulated to S_W" << std::endl;
            }
        }

        if (verbose) {
            PlaintextOps::printMatrix(Sw, f, f, "S_W", f_tilde);
        }

        // Store Sw and Sb for file saving
        result.Sw = Sw;
        result.Sb = Sb;

        // ========== Step 5: Compute S_W^{-1} ==========
        if (verbose) std::cout << "\n[Step 5] Computing S_W^{-1} (Schulz iteration, " << inversionIterations << " iters)..." << std::endl;

        // For inversion, we need to work with actual matrix dimension
        // Extract the f x f submatrix from the padded matrix
        std::vector<double> Sw_actual(f * f, 0.0);
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                Sw_actual[i * f + j] = Sw[i * f_tilde + j];
            }
        }

        auto Sw_inv_actual = invertMatrix(Sw_actual, f, inversionIterations);

        // Pad back to f_tilde x f_tilde
        std::vector<double> Sw_inv(f_tilde * f_tilde, 0.0);
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                Sw_inv[i * f_tilde + j] = Sw_inv_actual[i * f + j];
            }
        }

        if (verbose) {
            PlaintextOps::printMatrix(Sw_inv_actual, f, f, "S_W^{-1}");
        }

        // ========== Step 6: Compute S_W^{-1} * S_B ==========
        if (verbose) std::cout << "[Step 6] Computing S_W^{-1} * S_B..." << std::endl;

        // Extract Sb to actual dimensions
        std::vector<double> Sb_actual(f * f, 0.0);
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                Sb_actual[i * f + j] = Sb[i * f_tilde + j];
            }
        }

        auto Sw_inv_Sb = PlaintextOps::matMult(Sw_inv_actual, Sb_actual, f);
        result.Sw_inv_Sb = Sw_inv_Sb;
        result.Sw_inv = Sw_inv_actual;  // Store for direct Fisher direction computation

        if (verbose) {
            PlaintextOps::printMatrix(Sw_inv_Sb, f, f, "S_W^{-1} * S_B");
        }

        // ========== Step 7: Compute Eigenvector w = S_W^{-1} * (mu_1 - mu_0) ==========
        if (verbose) std::cout << "[Step 7] Computing eigenvector w = S_W^{-1} * (mu_1 - mu_0)..." << std::endl;

        // Compute diff = mu_1 - mu_0
        std::vector<double> diff(f);
        for (size_t i = 0; i < f; i++) {
            diff[i] = result.classMeans[1][i] - result.classMeans[0][i];
        }

        // w = S_W^{-1} * diff
        result.eigenvector.resize(f, 0.0);
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                result.eigenvector[i] += Sw_inv_actual[i * f + j] * diff[j];
            }
        }

        if (verbose) {
            PlaintextOps::printVector(result.eigenvector, f, "Eigenvector w");
        }

        // ========== Step 8: Compute Projected Means ==========
        if (verbose) std::cout << "[Step 8] Computing projected means..." << std::endl;

        result.projectedMean0 = 0.0;
        result.projectedMean1 = 0.0;
        for (size_t i = 0; i < f; i++) {
            result.projectedMean0 += result.eigenvector[i] * result.classMeans[0][i];
            result.projectedMean1 += result.eigenvector[i] * result.classMeans[1][i];
        }

        if (verbose) {
            std::cout << "proj_mu_0 = " << result.projectedMean0 << std::endl;
            std::cout << "proj_mu_1 = " << result.projectedMean1 << std::endl;
        }

        if (verbose) {
            std::cout << "\n========== Training Complete ==========" << std::endl;
        }

        // Note: file saving now happens from main after inference
        (void)outputFile;  // Mark as intentionally unused

        return result;
    }

    static void verifyInversion(const std::vector<double>& Sw,
                                const std::vector<double>& Sw_inv,
                                int d) {
        auto product = PlaintextOps::matMult(Sw, Sw_inv, d);

        std::cout << "=== Verification: S_W * S_W^{-1} ===" << std::endl;
        double maxError = 0.0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double error = std::abs(product[i * d + j] - expected);
                maxError = std::max(maxError, error);
            }
        }
        std::cout << "Max error from identity: " << maxError << std::endl << std::endl;
    }

    static void saveResultsToFile(const std::string& filename,
                                  const LDATrainResult& result,
                                  const std::vector<double>& Sw,
                                  const std::vector<double>& Sb,
                                  size_t f,
                                  size_t f_tilde,
                                  double accuracy = -1.0,
                                  int correct = 0,
                                  int total = 0,
                                  double precision = 0.0,
                                  double recall = 0.0,
                                  double f1 = 0.0,
                                  int trainSamples = 0) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

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

        // S_B skipped (not needed for binary Fisher LDA)

        // Intermediate results: X_bar_c and S_c per class
        for (size_t c = 0; c < result.X_bar_c.size(); c++) {
            size_t s_c = result.classCounts[c];

            file << "=== X_bar_c (class " << c << ", first 10 rows, " << f << " cols) ===" << std::endl;
            for (size_t row = 0; row < 10 && row < s_c; row++) {
                for (size_t col = 0; col < f; col++) {
                    file << std::setw(10) << result.X_bar_c[c][row * f_tilde + col] << " ";
                }
                file << std::endl;
            }
            file << std::endl;

            file << "=== S_c (class " << c << " scatter, " << f << "x" << f << ") ===" << std::endl;
            for (size_t i = 0; i < f; i++) {
                for (size_t j = 0; j < f; j++) {
                    file << std::setw(10) << result.S_c[c][i * f_tilde + j] << " ";
                }
                file << std::endl;
            }
            file << std::endl;
        }

        file << "=== S_W (" << f << "x" << f << ") ===" << std::endl;
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                file << std::setw(10) << Sw[i * f_tilde + j] << " ";
            }
            file << std::endl;
        }
        file << std::endl;

        file << "=== S_W^{-1} (" << f << "x" << f << ") ===" << std::endl;
        for (size_t i = 0; i < f; i++) {
            for (size_t j = 0; j < f; j++) {
                file << std::setw(10) << result.Sw_inv[i * f + j] << " ";
            }
            file << std::endl;
        }
        file << std::endl;

        // S_W^{-1} * S_B skipped (not needed for binary Fisher LDA)

        // Eigenvector (Fisher discriminant direction)
        file << "=== Eigenvector w = S_W^{-1} * (mu_1 - mu_0) (len=" << f << ") ===" << std::endl;
        for (size_t i = 0; i < f; i++) {
            file << std::setw(10) << result.eigenvector[i] << " ";
        }
        file << std::endl << std::endl;

        // Projected means
        file << "=== Projected Means ===" << std::endl;
        file << "proj_mu_0 = " << result.projectedMean0 << std::endl;
        file << "proj_mu_1 = " << result.projectedMean1 << std::endl;
        file << std::endl;

        // Accuracy and metrics
        if (accuracy >= 0) {
            file << "=== Training Info ===" << std::endl;
            file << "Training samples: " << trainSamples << std::endl;
            file << std::endl;

            file << "=== Inference Results ===" << std::endl;
            file << "Correct: " << correct << " / " << total << std::endl;
            file << "Accuracy: " << std::setprecision(2) << accuracy << "%" << std::endl;
            file << "Precision: " << std::setprecision(2) << precision << "%" << std::endl;
            file << "Recall: " << std::setprecision(2) << recall << "%" << std::endl;
            file << "F1 Score: " << std::setprecision(2) << f1 << "%" << std::endl;
            file << std::endl;
        }

        file.close();
        std::cout << "Results saved to: " << filename << std::endl;
    }
};
