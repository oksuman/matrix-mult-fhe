// main_encrypted.cpp
// Encrypted Logistic Regression (Fixed Hessian) with AR24 vs NewCol comparison
// CKKS homomorphic encryption

#include "lr_data_encoder.h"
#include "lr_he_trainer.h"
#include "lr_encrypted_ar24.h"
#include "lr_encrypted_newcol.h"
#include "encryption.h"
#include "../../common/evaluation_metrics.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

// Generate rotation indices (power-of-2 only)
std::vector<int> generateRotationIndices(int batchSize) {
    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    return rotations;
}

// Perform plaintext inference using decrypted weights
// Returns ClassificationResult for comparison
EvalMetrics::ClassificationResult performInference(const std::vector<double>& weights,
                                                    const LRDataset& testSet,
                                                    const std::string& methodName) {
    EvalMetrics::ClassificationResult result;
    result.total = testSet.numSamples;

    for (size_t i = 0; i < testSet.numSamples; i++) {
        double z = 0.0;
        for (int j = 0; j < LR_FEATURES; j++) {
            z += testSet.samples[i][j] * weights[j];
        }

        double pred_prob = 0.5 + z / 4.0;
        int pred_label = (pred_prob >= 0.5) ? 1 : -1;
        int true_label = testSet.labels[i];

        if (pred_label == true_label) result.correct++;
        if (pred_label == 1 && true_label == 1) result.tp++;
        else if (pred_label == 1 && true_label == -1) result.fp++;
        else if (pred_label == -1 && true_label == 1) result.fn++;
        else result.tn++;
    }

    result.print(methodName);
    return result;
}

// Result structure for algorithm comparison
struct FHExperimentResult {
    EvalMetrics::ClassificationResult classResult;
    std::chrono::duration<double> totalTime{0};
    bool valid = false;
};

// Run Fixed Hessian training with a given inversion algorithm
// Measures TOTAL time independently: from encrypted X,y to final weights
template<typename InvAlgorithm>
FHExperimentResult runFixedHessian(const std::string& algorithmName,
                     std::shared_ptr<Encryption> enc,
                     CryptoContext<DCRTPoly> cc,
                     KeyPair<DCRTPoly> keyPair,
                     const std::vector<int>& rotIndices,
                     int multDepth,
                     bool useBootstrapping,
                     const Ciphertext<DCRTPoly>& X_enc,
                     const Ciphertext<DCRTPoly>& y_enc,
                     const LRDataset& testSet,
                     const LRPrecomputed& pre,
                     int inversionIterations,
                     bool verbose) {
    using namespace std::chrono;
    FHExperimentResult expResult;

    EvalMetrics::printExperimentHeader("Fixed Hessian", algorithmName,
        LR_BATCH_SIZE, testSet.numSamples, LR_RAW_FEATURES);
    std::cout << "  Bootstrapping:    " << (useBootstrapping ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    InvAlgorithm algo(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);
    algo.setVerbose(verbose);

    const int f = LR_FEATURES;              // 16
    const int sampleDim = LR_MATRIX_DIM;    // 64
    const int batchSize = sampleDim * sampleDim;  // 4096

    // ========== Start TOTAL time measurement ==========
    auto totalStart = high_resolution_clock::now();

    // Step 1a: Compute X^T y
    std::cout << "\n[Step 1a] Computing X^T y..." << std::endl;
    auto xtyStart = high_resolution_clock::now();
    auto Xty_enc = algo.computeXty(X_enc, y_enc, f, sampleDim);
    auto xtyEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtyTime = xtyEnd - xtyStart;
    std::cout << "  X^T y time: " << std::fixed << std::setprecision(3) << xtyTime.count() << " s" << std::endl;

    // Verify X^T y
    if (verbose) {
        Plaintext ptxXty;
        cc->Decrypt(keyPair.secretKey, Xty_enc, &ptxXty);
        auto xtyVec = ptxXty->GetRealPackedValue();
        std::cout << "  [X^T y] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++) std::cout << std::setprecision(4) << xtyVec[i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.Xty[0][i] - xtyVec[i]));
        std::cout << "  [X^T y] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }
    Xty_enc->SetSlots(f * f);

    // Step 1b: Compute X^T (transpose)
    std::cout << "[Step 1b] Computing X^T (transpose)..." << std::endl;
    auto xtStart = high_resolution_clock::now();
    auto Xt_enc = algo.eval_transpose(X_enc, sampleDim, batchSize);
    auto xtEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtTime = xtEnd - xtStart;
    std::cout << "  X^T time: " << std::fixed << std::setprecision(3) << xtTime.count() << " s" << std::endl;

    // Step 1c: Compute X^T X (JKLS18)
    std::cout << "[Step 1c] Computing X^T X (JKLS18, " << sampleDim << "x" << sampleDim << ")..." << std::endl;
    auto xtxStart = high_resolution_clock::now();
    auto XtX_64 = algo.eval_mult_JKLS18(Xt_enc, X_enc, sampleDim);
    auto xtxEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtxTime = xtxEnd - xtxStart;
    std::cout << "  X^T X time: " << std::fixed << std::setprecision(3) << xtxTime.count() << " s" << std::endl;

    // Step 1d: Rebatch to 16x16
    std::cout << "[Step 1d] Rebatching X^T X from " << sampleDim << "x" << sampleDim << " to " << f << "x" << f << "..." << std::endl;
    auto rbStart = high_resolution_clock::now();
    auto XtX_16 = algo.rebatchToFeatureSpace(XtX_64, sampleDim, f);
    auto rbEnd = high_resolution_clock::now();
    std::chrono::duration<double> rbTime = rbEnd - rbStart;
    std::cout << "  Rebatch time: " << std::fixed << std::setprecision(3) << rbTime.count() << " s" << std::endl;

    // Verify X^T X diagonal
    if (verbose) {
        auto XtX_16_tmp = XtX_16->Clone();
        XtX_16_tmp->SetSlots(f * f);
        Plaintext ptxXtX;
        cc->Decrypt(keyPair.secretKey, XtX_16_tmp, &ptxXtX);
        auto xtxVec = ptxXtX->GetRealPackedValue();
        std::cout << "  [X^T X diag] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++)
            std::cout << std::setprecision(4) << xtxVec[i * f + i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.XtX[0][i * f + i] - xtxVec[i * f + i]));
        std::cout << "  [X^T X diag] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }

    auto precompTime = xtyTime + xtTime + xtxTime + rbTime;
    std::cout << "  Precomputation subtotal: " << std::fixed << std::setprecision(3)
              << precompTime.count() << " s" << std::endl;

    // Step 2: Invert XtX (which is PSD, directly invertible)
    // H^{-1} = -4 * (XtX)^{-1}
    std::cout << "\n[Step 2] Computing (X^T X)^{-1} with " << algorithmName << "..." << std::endl;
    auto invStart = high_resolution_clock::now();

    // XtX is in rebatched form (d*d*s slots from rebatchToFeatureSpace)
    // Need to get it back to d*d slots for inversion input
    auto XtX_for_inv = XtX_16->Clone();
    XtX_for_inv->SetSlots(f * f);

    // trace(XtX) upper bound: for min-max [0,1] normalized data
    // each feature in [0,1], XtX[i][i] <= N, trace <= N*16
    double traceUpperBound = (double)LR_BATCH_SIZE * 16;
    auto XtX_inv = algo.eval_inverse(XtX_for_inv, f, inversionIterations,
                                      LR_RAW_FEATURES + 1, traceUpperBound);
    // actualDim = 14 (13 features + 1 bias), padding at 14,15

    auto invEnd = high_resolution_clock::now();
    std::chrono::duration<double> invTime = invEnd - invStart;
    std::cout << "  Inversion time: " << std::fixed << std::setprecision(3)
              << invTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxInv;
        cc->Decrypt(keyPair.secretKey, XtX_inv, &ptxInv);
        auto invVec = ptxInv->GetRealPackedValue();
        std::cout << "  (X^T X)^{-1} diagonal: ";
        for (int i = 0; i < std::min(f, 8); i++) {
            std::cout << std::setprecision(6) << invVec[i * f + i] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Step 3: Compute gradient g = 0.5*Xty - 0.25*XtX*w_0
    std::cout << "[Step 3] Computing gradient..." << std::endl;
    auto gradStart = high_resolution_clock::now();

    // Initial weights: w_0 = [0.001, ..., 0.001, 0, 0] (14 active + 2 padding)
    std::vector<double> w0(f, 0.0);
    for (int j = 0; j <= LR_RAW_FEATURES; j++) {
        w0[j] = 0.001;
    }

    // Build column-replicated w_0 plaintext for mat-vec multiply
    // position(i,j) = w0[i] (after transpose: each column = w0)
    std::vector<double> w0_T(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w0_T[i * f + j] = w0[i];
        }
    }

    // XtX * w_0 using plaintext vector (saves 1 level)
    auto XtX_16_dd = XtX_16->Clone();
    XtX_16_dd->SetSlots(f * f);
    auto XtXw = algo.matVecMult_plain(XtX_16_dd, w0, f);

    // g = 0.5 * Xty - 0.25 * XtXw
    auto Xty_scaled = cc->EvalMult(Xty_enc, 0.5);
    auto XtXw_scaled = cc->EvalMult(XtXw, 0.25);
    auto g = cc->EvalSub(Xty_scaled, XtXw_scaled);

    if (verbose) {
        Plaintext ptxG;
        cc->Decrypt(keyPair.secretKey, g, &ptxG);
        auto gVec = ptxG->GetRealPackedValue();
        std::cout << "  Gradient (first 8): ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::setprecision(6) << gVec[i] << " ";
        }
        std::cout << std::endl;
    }

    // Step 4: delta_w = -4 * XtX_inv * g
    // w = w_0 - H^{-1} * g = w_0 - (-4 * XtX_inv * g) = w_0 + 4 * XtX_inv * g
    std::cout << "[Step 4] Computing weight update..." << std::endl;

    // Transpose g for mat-vec multiply: g -> column-replicated
    auto g_T = algo.eval_transpose(g, f, f * f);

    // XtX_inv * g_T (Hadamard + rowFoldingSum)
    auto inv_g = cc->EvalMultAndRelinearize(XtX_inv, g_T);

    // Row folding sum
    for (int i = (int)log2(f) - 1; i >= 0; i--) {
        int shift = f * (1 << i);
        cc->EvalAddInPlace(inv_g, cc->EvalRotate(inv_g, shift));
    }

    // delta_w = -4 * inv_g
    auto delta_w = cc->EvalMult(inv_g, -4.0);

    // w = w_0 - delta_w (note: delta_w = H^{-1}*g = -4*XtX_inv*g, so w = w_0 - (-4*XtX_inv*g))
    // Build w_0 as plaintext in 16x16 row-replicated form
    std::vector<double> w0_rep(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w0_rep[i * f + j] = w0[j];
        }
    }
    auto w0_ptx = cc->MakeCKKSPackedPlaintext(w0_rep, 1, 0, nullptr, f * f);
    auto w_enc = cc->EvalSub(w0_ptx, delta_w);

    auto gradEnd = high_resolution_clock::now();
    std::chrono::duration<double> gradTime = gradEnd - gradStart;
    std::cout << "  Gradient + update time: " << std::fixed << std::setprecision(3)
              << gradTime.count() << " s" << std::endl;

    // Step 5: Decrypt and infer
    std::cout << "[Step 5] Decrypting weights and running inference..." << std::endl;
    Plaintext ptxW;
    cc->Decrypt(keyPair.secretKey, w_enc, &ptxW);
    auto wVec = ptxW->GetRealPackedValue();

    // Extract first row (weights are row-replicated after folding)
    std::vector<double> weights(f, 0.0);
    for (int j = 0; j < f; j++) {
        weights[j] = wVec[j];
    }

    std::cout << "\n  Weights: ";
    for (int j = 0; j < LR_RAW_FEATURES; j++) {
        std::cout << std::setw(8) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << "\n  Bias (w[" << LR_RAW_FEATURES << "]): " << weights[LR_RAW_FEATURES] << std::endl;

    expResult.classResult = performInference(weights, testSet, algorithmName);

    // ========== End TOTAL time measurement ==========
    auto totalEnd = high_resolution_clock::now();
    expResult.totalTime = totalEnd - totalStart;
    expResult.valid = true;

    // Print timing summary using unified format
    EvalMetrics::TimingResult timing;
    timing.step1 = xtyTime + xtTime + xtxTime + rbTime;
    timing.step2 = invTime;
    timing.step3 = gradTime;
    timing.total = expResult.totalTime;
    timing.step1Name = "Precomputation";
    timing.step2Name = "Matrix inversion";
    timing.step3Name = "Gradient+update";
    timing.print(algorithmName);

    EvalMetrics::printExperimentFooter();

    return expResult;
}

// Run Simplified Fixed Hessian (diagonal inverse with Newton-Raphson, fully encrypted)
// Measures TOTAL time independently: from encrypted X,y to final weights
void runSimplifiedHessian(std::shared_ptr<Encryption> enc,
                          CryptoContext<DCRTPoly> cc,
                          KeyPair<DCRTPoly> keyPair,
                          const std::vector<int>& rotIndices,
                          int multDepth,
                          const Ciphertext<DCRTPoly>& X_enc,
                          const Ciphertext<DCRTPoly>& y_enc,
                          const LRDataset& testSet,
                          const LRPrecomputed& pre,  // For verification
                          int maxIter,               // Number of iterations
                          bool verbose) {
    using namespace std::chrono;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  Simplified Fixed Hessian (Newton-Raphson Diagonal Inverse)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int f = LR_FEATURES;              // 16
    const int sampleDim = LR_MATRIX_DIM;    // 64
    const int batchSize = sampleDim * sampleDim;  // 4096

    // Create a temporary LR_AR24 instance for helper functions
    LR_AR24 algo(enc, cc, keyPair, rotIndices, multDepth, false);
    algo.setVerbose(verbose);

    // ========== Start TOTAL time measurement ==========
    auto totalStart = high_resolution_clock::now();

    // ========== Step 1a: Compute X^T y ==========
    std::cout << "\n[Step 1a] Computing X^T y..." << std::endl;
    auto xtyStart = high_resolution_clock::now();
    auto Xty_enc = algo.computeXty(X_enc, y_enc, f, sampleDim);
    auto xtyEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtyTime = xtyEnd - xtyStart;
    std::cout << "  X^T y time: " << std::fixed << std::setprecision(3) << xtyTime.count() << " s" << std::endl;

    // Verify X^T y
    if (verbose) {
        Plaintext ptxXty;
        cc->Decrypt(keyPair.secretKey, Xty_enc, &ptxXty);
        auto xtyVec = ptxXty->GetRealPackedValue();
        std::cout << "  [X^T y] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++) std::cout << std::setprecision(4) << xtyVec[i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.Xty[0][i] - xtyVec[i]));
        std::cout << "  [X^T y] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }
    Xty_enc->SetSlots(f * f);

    // ========== Step 1b: Compute X^T (transpose) ==========
    std::cout << "[Step 1b] Computing X^T (transpose)..." << std::endl;
    auto xtStart = high_resolution_clock::now();
    auto Xt_enc = algo.eval_transpose(X_enc, sampleDim, batchSize);
    auto xtEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtTime = xtEnd - xtStart;
    std::cout << "  X^T time: " << std::fixed << std::setprecision(3) << xtTime.count() << " s" << std::endl;

    // ========== Step 1c: Compute X^T X (JKLS18) ==========
    std::cout << "[Step 1c] Computing X^T X (JKLS18, " << sampleDim << "x" << sampleDim << ")..." << std::endl;
    auto xtxStart = high_resolution_clock::now();
    auto XtX_64 = algo.eval_mult_JKLS18(Xt_enc, X_enc, sampleDim);
    auto xtxEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtxTime = xtxEnd - xtxStart;
    std::cout << "  X^T X time: " << std::fixed << std::setprecision(3) << xtxTime.count() << " s" << std::endl;

    // Debug: Check JKLS18 result before rebatch
    if (verbose) {
        Plaintext ptxXtX64;
        cc->Decrypt(keyPair.secretKey, XtX_64, &ptxXtX64);
        auto xtx64Vec = ptxXtX64->GetRealPackedValue();
        std::cout << "  [JKLS18 raw] XtX_64[0,0] = " << xtx64Vec[0]
                  << " (expected: " << pre.XtX[0][0] << ")" << std::endl;
        std::cout << "  [JKLS18 raw] XtX_64[1,1] = " << xtx64Vec[sampleDim + 1]
                  << " (expected: " << pre.XtX[0][f + 1] << ")" << std::endl;
    }

    // ========== Step 1d: Rebatch to 16x16 ==========
    std::cout << "[Step 1d] Rebatching X^T X from " << sampleDim << "x" << sampleDim << " to " << f << "x" << f << "..." << std::endl;
    auto rbStart = high_resolution_clock::now();
    auto XtX_16 = algo.rebatchToFeatureSpace(XtX_64, sampleDim, f);
    auto rbEnd = high_resolution_clock::now();
    std::chrono::duration<double> rbTime = rbEnd - rbStart;
    std::cout << "  Rebatch time: " << std::fixed << std::setprecision(3) << rbTime.count() << " s" << std::endl;

    // Verify X^T X diagonal
    if (verbose) {
        auto XtX_16_tmp = XtX_16->Clone();
        XtX_16_tmp->SetSlots(f * f);
        Plaintext ptxXtX;
        cc->Decrypt(keyPair.secretKey, XtX_16_tmp, &ptxXtX);
        auto xtxVec = ptxXtX->GetRealPackedValue();
        std::cout << "  [X^T X diag] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++)
            std::cout << std::setprecision(4) << xtxVec[i * f + i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.XtX[0][i * f + i] - xtxVec[i * f + i]));
        std::cout << "  [X^T X diag] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }

    auto precompTime = xtyTime + xtTime + xtxTime + rbTime;
    std::cout << "  Precomputation subtotal: " << std::fixed << std::setprecision(3)
              << precompTime.count() << " s" << std::endl;

    // ========== Step 2: Compute diagonal Hessian from X (encrypted) ==========
    std::cout << "\n[Step 2] Computing diagonal Hessian from X (encrypted)..." << std::endl;
    auto diagHStart = high_resolution_clock::now();

    auto diagH = algo.computeDiagHessian_enc(X_enc, sampleDim, f);

    auto diagHEnd = high_resolution_clock::now();
    std::chrono::duration<double> diagHTime = diagHEnd - diagHStart;
    std::cout << "  Diagonal Hessian time: " << std::fixed << std::setprecision(3)
              << diagHTime.count() << " s" << std::endl;

    // Verify diagonal Hessian against plaintext
    // Note: pre.inv_diag_H stores 1/H̃(j), so H̃(j) = -1/pre.inv_diag_H[j*f+j]
    {
        Plaintext ptxDiagH;
        cc->Decrypt(keyPair.secretKey, diagH, &ptxDiagH);
        auto diagHVec = ptxDiagH->GetRealPackedValue();

        std::cout << "\n  [diag(H̃)] Verification (first 14):" << std::endl;
        std::cout << "    Plaintext:  ";
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            // diag_H[j] = 1 / inv_diag_H[j]
            double diagH_plain = (std::abs(pre.inv_diag_H[i * f + i]) > 1e-10)
                                 ? (1.0 / pre.inv_diag_H[i * f + i])
                                 : 0.0;
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << diagH_plain << " ";
        }
        std::cout << std::endl;
        std::cout << "    Encrypted:  ";
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << diagHVec[i] << " ";
        }
        std::cout << std::endl;

        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            double diagH_plain = (std::abs(pre.inv_diag_H[i * f + i]) > 1e-10)
                                 ? (1.0 / pre.inv_diag_H[i * f + i])
                                 : 0.0;
            maxErr = std::max(maxErr, std::abs(diagH_plain - diagHVec[i]));
        }
        std::cout << "    Max error:  " << std::scientific << maxErr << std::endl;
    }

    // ========== Step 3: Newton-Raphson for diagonal inverse ==========
    std::cout << "[Step 3] Newton-Raphson diagonal inverse..." << std::endl;
    auto nrStart = high_resolution_clock::now();

    auto invDiag = algo.eval_diagonal_inverse_newton_raphson(diagH, LR_BATCH_SIZE, f, sampleDim, 8);

    auto nrEnd = high_resolution_clock::now();
    std::chrono::duration<double> nrTime = nrEnd - nrStart;
    std::cout << "  Newton-Raphson time: " << std::fixed << std::setprecision(3)
              << nrTime.count() << " s" << std::endl;

    // Verify Newton-Raphson inverse against plaintext
    {
        Plaintext ptxInvDiag;
        cc->Decrypt(keyPair.secretKey, invDiag, &ptxInvDiag);
        auto invVec = ptxInvDiag->GetRealPackedValue();

        std::cout << "\n  [diag(H̃^{-1})] Verification (first 14):" << std::endl;
        std::cout << "    Plaintext:  ";
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << pre.inv_diag_H[i * f + i] << " ";
        }
        std::cout << std::endl;
        std::cout << "    Encrypted:  ";
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << invVec[i] << " ";
        }
        std::cout << std::endl;

        double maxErr = 0;
        for (int i = 0; i < LR_RAW_FEATURES + 1; i++) {
            maxErr = std::max(maxErr, std::abs(pre.inv_diag_H[i * f + i] - invVec[i]));
        }
        std::cout << "    Max error:  " << std::scientific << maxErr << std::endl;

        // Document Newton-Raphson assumption (average-based)
        int actualFeatures = LR_RAW_FEATURES + 1;  // 14
        double u0 = -16.0 / (LR_BATCH_SIZE * actualFeatures);
        std::cout << "\n  [Newton-Raphson Assumptions]" << std::endl;
        std::cout << "    Initial guess: u₀ = -16/(N×(d+1)) = " << u0 << std::endl;
        std::cout << "    Where N=" << LR_BATCH_SIZE << " samples, d+1=" << actualFeatures << " features" << std::endl;
        std::cout << "    Assumption: Data normalized to [0,1], uniform distribution" << std::endl;
        std::cout << "    E[diag_H] ≈ -N×(d+1)/16 = " << (-LR_BATCH_SIZE * actualFeatures / 16.0) << std::endl;
        std::cout << "    Single iteration: inv = 2u₀ - a×u₀²" << std::endl;
    }

    // ========== Step 4: Iterative training ==========
    int bootstrapCount = 0;
    const int slots16 = f * f;  // 256 slots for 16-replicated form

    std::cout << "\n[Step 4] Iterative training (" << maxIter << " iterations)..." << std::endl;
    auto gradStart = high_resolution_clock::now();

    // Initial weights: 16-replicated form (256 slots)
    // [w0..w15, w0..w15, ...] repeated 16 times
    std::vector<double> w_vec(slots16, 0.0);
    for (int rep = 0; rep < f; rep++) {
        for (int j = 0; j <= LR_RAW_FEATURES; j++) {
            w_vec[rep * f + j] = 0.001;
        }
    }

    // Encrypt initial weights
    auto w_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(w_vec, 1, 0, nullptr, slots16));

    std::cout << "  Initial w level: " << w_enc->GetLevel() << "/" << multDepth << std::endl;

    // Precompute 0.5 * Xty in 16-replicated form (256 slots)
    // SetSlots automatically replicates to 16-replicated form, no folding needed
    Xty_enc->SetSlots(slots16);
    auto Xty_scaled = cc->EvalMult(Xty_enc, 0.5);

    // invDiag is already 16-replicated form (256 slots) from computeDiagHessian_enc

    auto XtX_16_dd = XtX_16->Clone();
    XtX_16_dd->SetSlots(slots16);

    for (int iter = 0; iter < maxIter; iter++) {
        // w is 16-replicated: [w0..w15, w0..w15, ...]
        // For XtX * w, need column-replicated: [w0*16, w1*16, ...]
        // Transpose: eval_transpose on 16x16
        auto w_col = algo.eval_transpose(w_enc, f, slots16);

        // XtXw = XtX * w (Hadamard + row folding)
        auto XtXw = cc->EvalMultAndRelinearize(XtX_16_dd, w_col);

        // Row folding sum
        for (int i = 0; i < (int)log2(f); i++) {
            int shift = f * (1 << i);
            cc->EvalAddInPlace(XtXw, cc->EvalRotate(XtXw, shift));
        }
        // XtXw is now 16-replicated (all rows same)

        // g = 0.5 * Xty - 0.25 * XtXw (both 16-replicated, 256 slots)
        auto XtXw_scaled = cc->EvalMult(XtXw, 0.25);
        auto g = cc->EvalSub(Xty_scaled, XtXw_scaled);

        // delta_w = invDiag * g (both 16-replicated, 256 slots)
        auto delta_w = cc->EvalMultAndRelinearize(invDiag, g);

        // Debug: print intermediate values for iter 1, 2
        if (iter < 2 && verbose) {
            Plaintext ptxXtXw, ptxG, ptxDeltaW;
            cc->Decrypt(keyPair.secretKey, XtXw_scaled, &ptxXtXw);
            cc->Decrypt(keyPair.secretKey, g, &ptxG);
            cc->Decrypt(keyPair.secretKey, delta_w, &ptxDeltaW);
            auto xtxwVec = ptxXtXw->GetRealPackedValue();
            auto gVec = ptxG->GetRealPackedValue();
            auto dwVec = ptxDeltaW->GetRealPackedValue();
            std::cout << "\n  [Debug iter " << (iter+1) << "] XtXw (first 8): ";
            for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << xtxwVec[j] << " ";
            std::cout << "\n  [Debug iter " << (iter+1) << "] g (first 8): ";
            for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << gVec[j] << " ";
            std::cout << "\n  [Debug iter " << (iter+1) << "] delta_w (first 8): ";
            for (int j = 0; j < 8; j++) std::cout << std::setprecision(4) << dwVec[j] << " ";
            std::cout << std::endl;
        }

        // w = w - delta_w (16-replicated)
        w_enc = cc->EvalSub(w_enc, delta_w);

        std::cout << "  [Iter " << (iter + 1) << "] w level: " << w_enc->GetLevel()
                  << "/" << multDepth;

        // Check if bootstrapping needed
        if ((int)w_enc->GetLevel() >= multDepth - 3) {
            std::cout << " -> Bootstrapping (pre-level: " << w_enc->GetLevel() << "/" << multDepth << ")...";
            // Reduce to 16 slots for bootstrap
            w_enc->SetSlots(f);
            w_enc = cc->EvalBootstrap(w_enc, 2);
            // Expand back to 16-replicated (256 slots)
            // SetSlots automatically replicates to 16-replicated form, no folding needed
            w_enc->SetSlots(slots16);
            bootstrapCount++;
            std::cout << " done (new level: " << w_enc->GetLevel() << ")";
        }

        // Check accuracy after this iteration
        {
            Plaintext ptxW;
            cc->Decrypt(keyPair.secretKey, w_enc, &ptxW);
            auto wVec = ptxW->GetRealPackedValue();
            std::vector<double> weights(f, 0.0);
            for (int j = 0; j < f; j++) weights[j] = wVec[j];

            int correct = 0;
            for (size_t i = 0; i < testSet.numSamples; i++) {
                double z = 0;
                for (int j = 0; j < f; j++) z += testSet.samples[i][j] * weights[j];
                int pred = (z >= 0) ? 1 : -1;
                if (pred == testSet.labels[i]) correct++;
            }
            std::cout << " | Acc: " << std::fixed << std::setprecision(2)
                      << (100.0 * correct / testSet.numSamples) << "%";
        }
        std::cout << std::endl;
    }

    auto gradEnd = high_resolution_clock::now();
    std::chrono::duration<double> gradTime = gradEnd - gradStart;

    std::cout << "  Total bootstraps: " << bootstrapCount << std::endl;

    // ========== Step 5: Decrypt and infer ==========
    std::cout << "\n[Step 5] Decrypting weights and inference..." << std::endl;

    Plaintext ptxW;
    cc->Decrypt(keyPair.secretKey, w_enc, &ptxW);
    auto wVec = ptxW->GetRealPackedValue();

    std::vector<double> weights(f, 0.0);
    for (int j = 0; j < f; j++) {
        weights[j] = wVec[j];
    }

    std::cout << "\n  Weights: ";
    for (int j = 0; j < LR_RAW_FEATURES; j++) {
        std::cout << std::setw(8) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << "\n  Bias (w[" << LR_RAW_FEATURES << "]): " << weights[LR_RAW_FEATURES] << std::endl;

    performInference(weights, testSet, "Simplified Fixed Hessian (Newton-Raphson)");

    // ========== End TOTAL time measurement ==========
    auto totalEnd = high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;

    std::cout << "\n--- Simplified Timing Summary ---" << std::endl;
    std::cout << "  X^T y:            " << std::setprecision(3) << xtyTime.count() << " s" << std::endl;
    std::cout << "  X^T:              " << xtTime.count() << " s" << std::endl;
    std::cout << "  X^T X:            " << xtxTime.count() << " s" << std::endl;
    std::cout << "  Rebatch:          " << rbTime.count() << " s" << std::endl;
    std::cout << "  Diagonal Hessian: " << diagHTime.count() << " s" << std::endl;
    std::cout << "  Newton-Raphson:   " << nrTime.count() << " s" << std::endl;
    std::cout << "  Gradient+update:  " << gradTime.count() << " s" << std::endl;
    std::cout << "  ---------------------------------" << std::endl;
    std::cout << "  TOTAL:            " << totalTime.count() << " s" << std::endl;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    bool verbose = true;
    bool useBootstrapping = true;
    std::string algorithm = "both";  // "ar24", "newcol", "both", "simplified"
    int inversionIterations = getFHInversionIterations(LR_FEATURES);  // For AR24/NewCol matrix inversion
#ifdef DATASET_DIABETES
    int simplifiedIterations = 256;  // Diabetes: slower convergence, needs more iterations
#else
    int simplifiedIterations = 8;    // Heart Disease: fast convergence
#endif

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            verbose = false;
        } else if (arg == "--no-bootstrap") {
            useBootstrapping = false;
        } else if (arg == "--ar24") {
            algorithm = "ar24";
        } else if (arg == "--newcol") {
            algorithm = "newcol";
        } else if (arg == "--simplified") {
            algorithm = "simplified";
        } else if (arg == "--iterations" && i + 1 < argc) {
            inversionIterations = std::stoi(argv[++i]);
        } else if (arg == "--simp-iter" && i + 1 < argc) {
            simplifiedIterations = std::stoi(argv[++i]);
        }
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    Logistic Regression (Fixed Hessian) - Encrypted Mode     #" << std::endl;
#ifdef DATASET_DIABETES
    std::cout << "#    Diabetes Dataset (8 features, 64 train samples)          #" << std::endl;
#else
    std::cout << "#    Heart Disease Dataset (13 features, 128 train samples)   #" << std::endl;
#endif
    std::cout << "#    AR24 vs NewCol Matrix Inversion Comparison               #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    // ========== Step 1: Load and preprocess data ==========
#ifdef DATASET_DIABETES
    std::string trainPath = std::string(DATA_DIR) + "/diabetes_train.csv";
    std::string testPath = std::string(DATA_DIR) + "/diabetes_test.csv";
#else
    std::string trainPath = (LR_BATCH_SIZE == 64)
        ? std::string(DATA_DIR) + "/heart_train_64.csv"
        : std::string(DATA_DIR) + "/heart_train_128.csv";
    std::string testPath = std::string(DATA_DIR) + "/heart_test.csv";
#endif

    std::cout << "\n[1] Loading datasets..." << std::endl;
    auto trainSet = LRDataEncoder::loadCSV(trainPath);
    auto testSet = LRDataEncoder::loadCSV(testPath);

    LRDataEncoder::printDatasetInfo(trainSet, "Raw Train");
    LRDataEncoder::printDatasetInfo(testSet, "Raw Test");

    std::cout << "[2] Min-max normalization [0,1]..." << std::endl;
    LRDataEncoder::normalizeFeatures(trainSet);
    LRDataEncoder::normalizeWithParams(testSet, trainSet);

    std::cout << "[3] Adding bias column and padding to 16 features..." << std::endl;
    LRDataEncoder::addBiasAndPad(trainSet);
    LRDataEncoder::addBiasAndPad(testSet);

    LRDataEncoder::printDatasetInfo(trainSet, "Processed Train");
    LRDataEncoder::printDatasetInfo(testSet, "Processed Test");

    // Plaintext precomputation (for simplified Hessian and verification)
    const int NUM_BATCHES = 1;
    auto pre = LRHETrainer::precompute(trainSet, NUM_BATCHES, verbose);

    // ========== Step 2: Setup CKKS Encryption ==========
    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;

    int multDepth;
    uint32_t scalingModSize, firstModSize;

    // Unified parameters (bootstrapping always enabled for benchmarks)
    multDepth = 28;
    scalingModSize = 59;
    firstModSize = 60;

    int sampleDim = LR_MATRIX_DIM;
    int batchSize = sampleDim * sampleDim;

    std::cout << "Sample dimension: " << sampleDim << "x" << sampleDim << std::endl;
    std::cout << "Feature dimension: " << LR_FEATURES << "x" << LR_FEATURES << std::endl;
    std::cout << "Multiplicative depth: " << multDepth << std::endl;
    std::cout << "Bootstrapping: " << (useBootstrapping ? "enabled" : "disabled") << std::endl;
    std::cout << "Inversion iterations: " << inversionIterations << std::endl;

    auto rotIndices = generateRotationIndices(batchSize);
    std::cout << "Rotation indices: " << rotIndices.size() << std::endl;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    if (useBootstrapping) {
        cc->Enable(FHE);
    }

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    std::cout << "Generating rotation keys..." << std::flush;
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);
    std::cout << " Done." << std::endl;

    if (useBootstrapping) {
        std::cout << "Setting up bootstrapping..." << std::flush;
        std::vector<uint32_t> levelBudget = {4, 4};
        std::vector<uint32_t> bsgsDim = {0, 0};
        // Setup for both 256 slots (Full Fixed) and 16 slots (Simplified)
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, LR_FEATURES * LR_FEATURES);  // 256 slots
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, LR_FEATURES);                // 16 slots
        std::cout << " Setup done. Generating keys..." << std::flush;
        cc->EvalBootstrapKeyGen(keyPair.secretKey, LR_FEATURES * LR_FEATURES);    // 256 slots
        cc->EvalBootstrapKeyGen(keyPair.secretKey, LR_FEATURES);                  // 16 slots
        std::cout << " Done." << std::endl;
    }

    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);

    // ========== Step 3: Encrypt data ==========
    std::cout << "\n--- Encrypting Training Data ---" << std::endl;

    auto X_packed = LRDataEncoder::packBatchX(trainSet, 0);
    auto y_packed = LRDataEncoder::packBatchY(trainSet, 0);

    auto X_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(X_packed, 1, 0, nullptr, batchSize));
    auto y_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(y_packed, 1, 0, nullptr, sampleDim));

    std::cout << "  X encrypted: " << batchSize << " slots" << std::endl;
    std::cout << "  y encrypted: " << sampleDim << " slots" << std::endl;

    // ========== Step 4: Run each algorithm independently ==========
    // Each algorithm measures its TOTAL time from encrypted X,y to final weights
    // No shared precomputation - each algorithm computes everything independently

    FHExperimentResult ar24Result, newcolResult;

    // ========== Run Fixed Hessian with AR24 ==========
    if (algorithm == "ar24" || algorithm == "both") {
        ar24Result = runFixedHessian<LR_AR24>(
            "AR24", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            X_enc, y_enc, testSet, pre, inversionIterations, verbose);
    }

    // ========== Run Fixed Hessian with NewCol ==========
    if (algorithm == "newcol" || algorithm == "both") {
        newcolResult = runFixedHessian<LR_NewCol>(
            "NewCol", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            X_enc, y_enc, testSet, pre, inversionIterations, verbose);
    }

    // ========== Run Simplified Fixed Hessian ==========
    if (algorithm == "simplified") {
        runSimplifiedHessian(enc, cc, keyPair, rotIndices, multDepth,
                             X_enc, y_enc, testSet, pre, simplifiedIterations, verbose);
    }

    // Print comparison summary if both algorithms were run
    if (ar24Result.valid && newcolResult.valid) {
        EvalMetrics::TimingResult ar24Timing, newcolTiming;
        ar24Timing.total = ar24Result.totalTime;
        newcolTiming.total = newcolResult.totalTime;

        EvalMetrics::printComparisonSummary(
            "Fixed Hessian",
            ar24Timing, newcolTiming,
            ar24Result.classResult.f1Score(),
            newcolResult.classResult.f1Score(),
            "F1 Score");
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
