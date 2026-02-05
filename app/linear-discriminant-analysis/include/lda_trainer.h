#pragma once

#include "lda_data_encoder.h"
#include "lda_plaintext_ops.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

struct LDATrainResult {
    std::vector<double> Sw_inv_Sb;      // S_W^{-1} * S_B matrix (d x d)
    std::vector<double> Sw_inv;         // S_W^{-1} matrix (for direct Fisher computation)
    std::vector<std::vector<double>> classMeans;  // Per-class mean vectors
    std::vector<double> globalMean;     // Global mean vector
    std::vector<size_t> classCounts;    // Number of samples per class (for threshold)

    size_t matrixDim;   // Actual feature dimension
    size_t paddedDim;   // Padded dimension (power of 2)
};

class LDATrainer {
private:
    // Schulz iteration for matrix inversion: Y_{i+1} = Y_i * (2I - A * Y_i)
    // Converges if spectral radius of (I - A*Y_0) < 1
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
    // Train LDA using CKKS-friendly operations
    static LDATrainResult train(const EncodedData& encoded,
                                const LDADataset& dataset,
                                int inversionIterations = 20,
                                bool verbose = false) {
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

        // ========== Step 3: Compute S_W (Within-class scatter) ==========
        // S_W = sum_c sum_{x in class c} (x - mu_c)(x - mu_c)^T
        //     = sum_c (X_c - mu_c)^T * (X_c - mu_c)
        if (verbose) std::cout << "[Step 3] Computing S_W (within-class scatter)..." << std::endl;

        std::vector<double> Sw(f_tilde * f_tilde, 0.0);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            // Replicate class mean to match encoded data shape
            auto meanReplicated = PlaintextOps::replicateRow(
                PlaintextOps::extractFirstRow(classMeanReplicated[c], f_tilde),
                s_tilde_c,
                f_tilde
            );

            // X_bar_c = X_c - mu_c (broadcast subtraction)
            auto X_bar_c = PlaintextOps::sub(encoded.classSamples[c], meanReplicated);

            // S_c = X_bar_c^T * X_bar_c
            auto S_c = PlaintextOps::computeXtX(X_bar_c, s_c, f, s_tilde_c, f_tilde);

            // Accumulate
            PlaintextOps::addInPlace(Sw, S_c);
        }

        if (verbose) {
            PlaintextOps::printMatrix(Sw, f, f, "S_W");
        }

        // ========== Step 4: Compute S_B (Between-class scatter) ==========
        // S_B = sum_c s_c * (mu_c - mu)(mu_c - mu)^T
        if (verbose) std::cout << "[Step 4] Computing S_B (between-class scatter)..." << std::endl;

        std::vector<double> Sb(f_tilde * f_tilde, 0.0);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];

            // diff = mu_c - mu
            auto classMean = PlaintextOps::extractFirstRow(classMeanReplicated[c], f_tilde);
            auto diff = PlaintextOps::sub(classMean, globalMean);

            // outer = diff * diff^T
            auto outer = PlaintextOps::outerProduct(diff, diff, f, f_tilde);

            // Scale by class size and accumulate
            auto scaled = PlaintextOps::multScalar(outer, static_cast<double>(s_c));
            PlaintextOps::addInPlace(Sb, scaled);
        }

        if (verbose) {
            PlaintextOps::printMatrix(Sb, f, f, "S_B");
        }

        // ========== Step 5: Compute S_W^{-1} ==========
        if (verbose) std::cout << "[Step 5] Computing S_W^{-1} (Schulz iteration, " << inversionIterations << " iters)..." << std::endl;

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

        if (verbose) {
            std::cout << "\n========== Training Complete ==========" << std::endl;
        }

        return result;
    }

    // Verify S_W inversion by computing S_W * S_W^{-1}
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
};
