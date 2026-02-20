#pragma once

#include "fh_data_encoder.h"
#include "fh_he_ops.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

struct FHTrainResult {
    std::vector<double> weights;  // length 16 (includes bias at index 13, padding at 14-15)
    int iterations;
    std::string method;  // "simplified" or "fixed"
};

struct FHPrecomputed {
    // Per-batch precomputed values
    std::vector<std::vector<double>> Xty;       // [batch] -> length 16 vector
    std::vector<std::vector<double>> XtX;       // [batch] -> 16x16 matrix (256 elements)

    // Simplified Hessian (16x16 diagonal matrix)
    std::vector<double> inv_diag_H; // 16x16 diagonal matrix (256 elements)

    // Fixed Hessian (full 16x16 matrix)
    std::vector<double> H;          // 16x16 matrix (256 elements)
    std::vector<double> H_inv;      // 16x16 matrix (256 elements)

    int numBatches;
    int N;  // total training samples
};

class FHHETrainer {
public:
    // ===== HE-friendly precomputation (64x64 format) =====

    // X^T y: y->SetSlots(64->4096)->transpose->Hadamard(X,y_t)->rowFoldingSum
    static std::vector<double> computeXty_HE(const std::vector<double>& X_packed,
                                              const std::vector<double>& y_packed) {
        const int d = FH_MATRIX_DIM;
        const int f = FH_FEATURES;

        // y (64 slots) -> SetSlots(64->4096): replicate y into 64x64 format
        auto y_replicated = FHOps::setSlots(y_packed, FH_SLOTS);

        // Transpose: position(i,j) = y[i] for all j
        auto y_transposed = FHOps::transpose(y_replicated, d);

        // Hadamard(X, y_transposed): position(i,j) = x_ij * y_i
        auto xy = FHOps::mult(X_packed, y_transposed);

        // Row folding sum: position(0,j) = sum_i x_ij * y_i = (X^T y)_j
        auto folded = FHOps::rowFoldingSum(xy, d);

        return std::vector<double>(folded.begin(), folded.begin() + f);
    }

    // X^T X: transpose(X)->matMult->rebatch(64->16)
    static std::vector<double> computeXtX_HE(const std::vector<double>& X_packed) {
        const int d = FH_MATRIX_DIM;
        const int f = FH_FEATURES;

        auto Xt = FHOps::transpose(X_packed, d);
        auto XtX_64 = FHOps::matMult(Xt, X_packed, d);
        return FHOps::rebatch(XtX_64, d, f);
    }

    // Simplified Hessian diagonal per batch:
    // columnFoldingSum->maskCol0->replicateCol0->Hadamard(X,rep)->rowFoldingSum
    static std::vector<double> computeDiagH_batch_HE(const std::vector<double>& X_packed) {
        const int d = FH_MATRIX_DIM;
        const int f = FH_FEATURES;

        // 1. Column folding sum (rotate +32,+16,+8,+4,+2,+1)
        auto colFolded = FHOps::columnFoldingSum(X_packed, d);

        // 2. Mask column 0
        auto masked = FHOps::maskColumn0(colFolded, d);

        // 3. Replicate column 0 to all columns
        auto replicated = FHOps::replicateColumn0(masked, d);

        // 4. Hadamard(X, replicated): position(i,j) = x_ij * sum_k x_ik
        auto product = FHOps::mult(X_packed, replicated);

        // 5. Row folding sum: position(0,j) = sum_i x_ij * sum_k x_ik
        auto folded = FHOps::rowFoldingSum(product, d);

        return std::vector<double>(folded.begin(), folded.begin() + f);
    }

    // 16x16 matrix-vector multiply (HE-friendly pattern)
    // Computes M^T * v (= M * v when M is symmetric)
    // SetSlots(v)->transpose->Hadamard(M,v_t)->rowFoldingSum
    static std::vector<double> matVecMult_HE16(const std::vector<double>& M_16x16,
                                                const std::vector<double>& v_16) {
        const int f = FH_FEATURES;

        // 1. v (16 slots) -> SetSlots(16->256): 16x replicate
        auto v_rep = FHOps::setSlots(v_16, f * f);

        // 2. Transpose (16x16)
        auto v_t = FHOps::transpose(v_rep, f);

        // 3. Hadamard(M, v_transposed)
        auto product = FHOps::mult(M_16x16, v_t);

        // 4. Row folding sum
        auto folded = FHOps::rowFoldingSum(product, f);

        // 5. Extract first 16 elements
        return std::vector<double>(folded.begin(), folded.begin() + f);
    }

    // ===== Main precomputation =====

    static FHPrecomputed precompute(const FHDataset& trainSet, int numBatches,
                                    bool verbose = false) {
        FHPrecomputed pre;
        pre.numBatches = numBatches;
        pre.N = numBatches * FH_BATCH_SIZE;

        const int f = FH_FEATURES;

        pre.Xty.resize(numBatches);
        pre.XtX.resize(numBatches);

        if (verbose) {
            std::cout << "\n========== Precomputation ==========" << std::endl;
            std::cout << "N = " << pre.N << ", batches = " << numBatches << std::endl;
        }

        std::vector<double> total_diag_h(f, 0.0);
        std::vector<double> total_XtX(f * f, 0.0);

        for (int b = 0; b < numBatches; b++) {
            auto X_packed = FHDataEncoder::packBatchX(trainSet, b);
            auto y_packed = FHDataEncoder::packBatchY(trainSet, b);

            pre.Xty[b] = computeXty_HE(X_packed, y_packed);
            pre.XtX[b] = computeXtX_HE(X_packed);

            auto diag_h_b = computeDiagH_batch_HE(X_packed);
            for (int j = 0; j < f; j++) {
                total_diag_h[j] += diag_h_b[j];
            }
            FHOps::addInPlace(total_XtX, pre.XtX[b]);

            if (verbose) {
                std::cout << "  Batch " << b << " done." << std::endl;
            }
        }

        // Simplified Hessian: inv_diag_H as 16x16 diagonal matrix
        // diag_H[j] = -1/4 * total_diag_h[j], inv = 1/diag_H[j]
        pre.inv_diag_H.resize(f * f, 0.0);
        for (int j = 0; j < f; j++) {
            double d_val = -0.25 * total_diag_h[j];
            if (std::abs(d_val) > 1e-10) {
                pre.inv_diag_H[j * f + j] = 1.0 / d_val;
            }
        }

        // Fixed Hessian: H = -1/4 * sum_b XtX_b (16x16)
        pre.H.resize(f * f);
        for (int i = 0; i < f * f; i++) {
            pre.H[i] = -0.25 * total_XtX[i];
        }
        pre.H_inv = FHOps::invertMatrix(pre.H, f, 20);

        if (verbose) {
            // Print X^T y
            std::cout << "\nX^T y (first 14): ";
            for (int j = 0; j < FH_RAW_FEATURES + 1; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << pre.Xty[0][j] << " ";
            }
            std::cout << std::endl;

            // Print X^T X diagonal
            std::cout << "diag(X^T X) (first 14): ";
            for (int j = 0; j < FH_RAW_FEATURES + 1; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << pre.XtX[0][j * f + j] << " ";
            }
            std::cout << std::endl;

            // Print diag_H (before -1/4)
            std::cout << "raw_diag_h (first 14): ";
            for (int j = 0; j < FH_RAW_FEATURES + 1; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << total_diag_h[j] << " ";
            }
            std::cout << std::endl;

            // Print diag_H (after -1/4)
            std::cout << "diag(H̃) = -1/4 * raw (first 14): ";
            for (int j = 0; j < FH_RAW_FEATURES + 1; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << (-0.25 * total_diag_h[j]) << " ";
            }
            std::cout << std::endl;

            std::cout << "diag(inv_diag_H): ";
            for (int j = 0; j < f; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << pre.inv_diag_H[j * f + j] << " ";
            }
            std::cout << std::endl;

            FHOps::printMatrix(pre.H, f, f, "H (16x16)");
            FHOps::printMatrix(pre.H_inv, f, f, "H_inv (16x16)");
        }

        return pre;
    }

    // ===== Training: per-batch cycling =====

    static FHTrainResult trainSimplified(const FHPrecomputed& pre, int maxIter,
                                         bool verbose = false) {
        FHTrainResult result;
        result.method = "simplified";
        result.iterations = maxIter;

        const int f = FH_FEATURES;

        // Initialize weights (small values for active features, 0 for padding)
        std::vector<double> w(f, 0.0);
        for (int j = 0; j <= FH_RAW_FEATURES; j++) {  // 0..13 (features + bias)
            w[j] = 0.001;
        }

        if (verbose) {
            std::cout << "\n========== Training: Simplified Fixed Hessian ==========" << std::endl;
            std::cout << "Iterations: " << maxIter << std::endl;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            int batchIdx = iter % pre.numBatches;

            // XtXw = XtX[batch] * w (16x16 mat-vec, HE-friendly)
            auto XtXw = matVecMult_HE16(pre.XtX[batchIdx], w);

            // g = 0.5 * Xty[batch] - 0.25 * XtXw
            std::vector<double> g(f);
            for (int j = 0; j < f; j++) {
                g[j] = 0.5 * pre.Xty[batchIdx][j] - 0.25 * XtXw[j];
            }

            // delta_w = inv_diag_H * g (diagonal mat-vec, HE-friendly)
            auto delta_w = matVecMult_HE16(pre.inv_diag_H, g);

            // Debug: print intermediate values for iter 1, 2
            if (verbose && iter < 2) {
                std::cout << "\n  [Plaintext iter " << (iter+1) << "] 0.25*XtXw (first 8): ";
                for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << (0.25 * XtXw[j]) << " ";
                std::cout << "\n  [Plaintext iter " << (iter+1) << "] g (first 8): ";
                for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << g[j] << " ";
                std::cout << "\n  [Plaintext iter " << (iter+1) << "] delta_w (first 8): ";
                for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << delta_w[j] << " ";
                std::cout << std::endl;
            }

            // w -= delta_w
            for (int j = 0; j < f; j++) {
                w[j] -= delta_w[j];
            }

            if (verbose && ((iter + 1) % 32 == 0 || iter == 0)) {
                std::cout << "  iter " << (iter + 1) << ": w[0]=" << w[0]
                          << " bias=" << w[FH_RAW_FEATURES] << std::endl;
            }
        }

        result.weights = w;
        return result;
    }

    static FHTrainResult trainFixed(const FHPrecomputed& pre, int maxIter,
                                    bool verbose = false) {
        FHTrainResult result;
        result.method = "fixed";
        result.iterations = maxIter;

        const int f = FH_FEATURES;

        std::vector<double> w(f, 0.0);
        for (int j = 0; j <= FH_RAW_FEATURES; j++) {
            w[j] = 0.001;
        }

        if (verbose) {
            std::cout << "\n========== Training: Fixed Hessian ==========" << std::endl;
            std::cout << "Iterations: " << maxIter << std::endl;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            int batchIdx = iter % pre.numBatches;

            // XtXw = XtX[batch] * w
            auto XtXw = matVecMult_HE16(pre.XtX[batchIdx], w);

            // g = 0.5 * Xty[batch] - 0.25 * XtXw
            std::vector<double> g(f);
            for (int j = 0; j < f; j++) {
                g[j] = 0.5 * pre.Xty[batchIdx][j] - 0.25 * XtXw[j];
            }

            // delta_w = H_inv * g (full mat-vec, HE-friendly)
            auto delta_w = matVecMult_HE16(pre.H_inv, g);

            // w -= delta_w
            for (int j = 0; j < f; j++) {
                w[j] -= delta_w[j];
            }

            if (verbose && ((iter + 1) % 4 == 0 || iter == 0)) {
                std::cout << "  iter " << (iter + 1) << ": w[0]=" << w[0]
                          << " bias=" << w[FH_RAW_FEATURES] << std::endl;
            }
        }

        result.weights = w;
        return result;
    }

    // ===== Inference =====

    struct InferenceResult {
        int correct;
        int total;
        double accuracy;
        double precision;
        double recall;
        double f1;
    };

    static InferenceResult inference(const FHTrainResult& model,
                                     const FHDataset& testSet,
                                     bool verbose = false) {
        InferenceResult res;
        res.total = testSet.numSamples;
        res.correct = 0;

        int tp = 0, fp = 0, fn = 0, tn = 0;

        for (size_t i = 0; i < testSet.numSamples; i++) {
            // z = x * w (dot product, x includes bias column and padding)
            double z = 0.0;
            for (int j = 0; j < FH_FEATURES; j++) {
                z += testSet.samples[i][j] * model.weights[j];
            }

            // sigma(z) ~ 1/2 + z/4
            double pred_prob = 0.5 + z / 4.0;

            // pred >= 0.5 -> +1, else -1
            int pred_label = (pred_prob >= 0.5) ? 1 : -1;
            int true_label = testSet.labels[i];

            if (pred_label == true_label) res.correct++;

            if (pred_label == 1 && true_label == 1) tp++;
            else if (pred_label == 1 && true_label == -1) fp++;
            else if (pred_label == -1 && true_label == 1) fn++;
            else tn++;
        }

        res.accuracy = 100.0 * res.correct / res.total;
        res.precision = (tp + fp > 0) ? 100.0 * tp / (tp + fp) : 0.0;
        res.recall = (tp + fn > 0) ? 100.0 * tp / (tp + fn) : 0.0;
        res.f1 = (res.precision + res.recall > 0)
                     ? 2.0 * res.precision * res.recall / (res.precision + res.recall)
                     : 0.0;

        if (verbose) {
            std::cout << "  Correct: " << res.correct << " / " << res.total << std::endl;
            std::cout << "  Accuracy:  " << std::fixed << std::setprecision(2) << res.accuracy << "%" << std::endl;
            std::cout << "  Precision: " << res.precision << "%" << std::endl;
            std::cout << "  Recall:    " << res.recall << "%" << std::endl;
            std::cout << "  F1 Score:  " << res.f1 << "%" << std::endl;
            std::cout << "  (TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn << ")" << std::endl;
        }

        return res;
    }
};
