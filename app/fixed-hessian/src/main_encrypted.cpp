#include "fh_data_encoder.h"
#include "fh_he_trainer.h"
#include "fh_encrypted_ar24.h"
#include "fh_encrypted_newcol.h"
#include "encryption.h"
#include "../../common/evaluation_metrics.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

std::vector<int> generateRotationIndices(int batchSize) {
    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    return rotations;
}

EvalMetrics::ClassificationResult performInference(const std::vector<double>& weights,
                                                    const FHDataset& testSet,
                                                    const std::string& methodName) {
    EvalMetrics::ClassificationResult result;
    result.total = testSet.numSamples;

    for (size_t i = 0; i < testSet.numSamples; i++) {
        double z = 0.0;
        for (int j = 0; j < FH_FEATURES; j++) {
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

struct FHExperimentResult {
    EvalMetrics::ClassificationResult classResult;
    std::chrono::duration<double> totalTime{0};
    bool valid = false;
};

struct SFHCheckpointResult {
    int iterations;
    double timeSec;
    double totalTimeSec;
    EvalMetrics::ClassificationResult classResult;
};

void saveFHResults(const std::string& filename,
                   const FHExperimentResult& ar24Result,
                   const FHExperimentResult& newcolResult,
                   const std::vector<SFHCheckpointResult>& sfhResults,
                   int numTestSamples) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    file << "================================================================\n";
    file << "  FH Benchmark Results\n";
    file << "  Generated: " << std::ctime(&time);
    file << "================================================================\n\n";

    file << "--- Configuration ---\n";
    file << "Training samples: " << FH_BATCH_SIZE << "\n";
    file << "Test samples:     " << numTestSamples << "\n";
    file << "Features (raw):   " << FH_RAW_FEATURES << "\n";
    file << "Trials:           1\n\n";

    file << std::fixed;

    file << "================================================================\n";
    file << "  TIMING & ACCURACY COMPARISON\n";
    file << "================================================================\n\n";

    file << std::left  << std::setw(22) << "  Method"
         << " | " << std::right << std::setw(4) << "Iter"
         << " | " << std::setw(9) << "Time (s)"
         << " | " << std::setw(9) << "Accuracy"
         << " | " << std::setw(8) << "F1 Score" << "\n";
    file << "  " << std::string(60, '-') << "\n";

    if (ar24Result.valid) {
        file << std::left  << std::setw(22) << "  FH (AR24)"
             << " | " << std::right << std::setw(4) << 1
             << " | " << std::setw(9) << std::setprecision(2) << ar24Result.totalTime.count()
             << " | " << std::setw(8) << ar24Result.classResult.accuracy() << "%"
             << " | " << std::setw(7) << ar24Result.classResult.f1Score() << "%\n";
    }

    if (newcolResult.valid) {
        file << std::left  << std::setw(22) << "  FH (NewCol)"
             << " | " << std::right << std::setw(4) << 1
             << " | " << std::setw(9) << std::setprecision(2) << newcolResult.totalTime.count()
             << " | " << std::setw(8) << newcolResult.classResult.accuracy() << "%"
             << " | " << std::setw(7) << newcolResult.classResult.f1Score() << "%\n";
    }

    for (const auto& cp : sfhResults) {
        file << std::left  << std::setw(22) << "  SFH"
             << " | " << std::right << std::setw(4) << cp.iterations
             << " | " << std::setw(9) << std::setprecision(2) << cp.totalTimeSec
             << " | " << std::setw(8) << cp.classResult.accuracy() << "%"
             << " | " << std::setw(7) << cp.classResult.f1Score() << "%\n";
    }

    file << "  " << std::string(62, '=') << "\n";
    file << "\n================================================================\n";
    file << "  END OF REPORT\n";
    file << "================================================================\n";

    file.close();
    std::cout << "\nResults saved to: " << filename << std::endl;
}

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
                     const FHDataset& testSet,
                     const FHPrecomputed& pre,
                     int inversionIterations,
                     bool verbose) {
    using namespace std::chrono;
    FHExperimentResult expResult;

    EvalMetrics::printExperimentHeader("Fixed Hessian", algorithmName,
        FH_BATCH_SIZE, testSet.numSamples, FH_RAW_FEATURES);
    std::cout << "  Bootstrapping:    " << (useBootstrapping ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    InvAlgorithm algo(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);
    algo.setVerbose(verbose);

    const int f = FH_FEATURES;
    const int sampleDim = FH_MATRIX_DIM;
    const int batchSize = sampleDim * sampleDim;

    auto totalStart = high_resolution_clock::now();

    std::cout << "\n[Step 1a] Computing X^T y..." << std::endl;
    auto xtyStart = high_resolution_clock::now();
    auto Xty_enc = algo.computeXty(X_enc, y_enc, f, sampleDim);
    auto xtyEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtyTime = xtyEnd - xtyStart;
    std::cout << "  X^T y time: " << std::fixed << std::setprecision(3) << xtyTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxXty;
        cc->Decrypt(keyPair.secretKey, Xty_enc, &ptxXty);
        auto xtyVec = ptxXty->GetRealPackedValue();
        std::cout << "  [X^T y] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++) std::cout << std::setprecision(4) << xtyVec[i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.Xty[0][i] - xtyVec[i]));
        std::cout << "  [X^T y] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }
    Xty_enc->SetSlots(f * f);
    std::cout << "[Step 1b] Computing X^T (transpose)..." << std::endl;
    auto xtStart = high_resolution_clock::now();
    auto Xt_enc = algo.eval_transpose(X_enc, sampleDim, batchSize);
    auto xtEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtTime = xtEnd - xtStart;
    std::cout << "  X^T time: " << std::fixed << std::setprecision(3) << xtTime.count() << " s" << std::endl;

    std::cout << "[Step 1c] Computing X^T X (JKLS18, " << sampleDim << "x" << sampleDim << ")..." << std::endl;
    auto xtxStart = high_resolution_clock::now();
    auto XtX_64 = algo.eval_mult_JKLS18(Xt_enc, X_enc, sampleDim);
    auto xtxEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtxTime = xtxEnd - xtxStart;
    std::cout << "  X^T X time: " << std::fixed << std::setprecision(3) << xtxTime.count() << " s" << std::endl;

    std::cout << "[Step 1d] Rebatching X^T X from " << sampleDim << "x" << sampleDim << " to " << f << "x" << f << "..." << std::endl;
    auto rbStart = high_resolution_clock::now();
    auto XtX_16 = algo.rebatchToFeatureSpace(XtX_64, sampleDim, f);
    auto rbEnd = high_resolution_clock::now();
    std::chrono::duration<double> rbTime = rbEnd - rbStart;
    std::cout << "  Rebatch time: " << std::fixed << std::setprecision(3) << rbTime.count() << " s" << std::endl;

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
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.XtX[0][i * f + i] - xtxVec[i * f + i]));
        std::cout << "  [X^T X diag] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }

    auto precompTime = xtyTime + xtTime + xtxTime + rbTime;
    std::cout << "  Precomputation subtotal: " << std::fixed << std::setprecision(3)
              << precompTime.count() << " s" << std::endl;

    std::cout << "\n[Step 2] Computing (X^T X)^{-1} with " << algorithmName << "..." << std::endl;
    auto invStart = high_resolution_clock::now();

    auto XtX_for_inv = XtX_16->Clone();
    XtX_for_inv->SetSlots(f * f);

    double traceUpperBound = (double)FH_BATCH_SIZE * 16;
    auto XtX_inv = algo.eval_inverse(XtX_for_inv, f, inversionIterations,
                                      FH_RAW_FEATURES + 1, traceUpperBound);

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

    // g = 0.5*Xty - 0.25*XtX*w_0
    std::cout << "[Step 3] Computing gradient..." << std::endl;
    auto gradStart = high_resolution_clock::now();

    std::vector<double> w0(f, 0.0);
    for (int j = 0; j <= FH_RAW_FEATURES; j++) {
        w0[j] = 0.001;
    }

    std::vector<double> w0_T(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w0_T[i * f + j] = w0[i];
        }
    }

    auto XtX_16_dd = XtX_16->Clone();
    XtX_16_dd->SetSlots(f * f);
    auto XtXw = algo.matVecMult_plain(XtX_16_dd, w0, f);

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

    // delta_w = -4 * XtX_inv * g, w = w_0 - delta_w
    std::cout << "[Step 4] Computing weight update..." << std::endl;

    auto g_T = algo.eval_transpose(g, f, f * f);
    auto inv_g = cc->EvalMultAndRelinearize(XtX_inv, g_T);

    for (int i = (int)log2(f) - 1; i >= 0; i--) {
        int shift = f * (1 << i);
        cc->EvalAddInPlace(inv_g, cc->EvalRotate(inv_g, shift));
    }

    auto delta_w = cc->EvalMult(inv_g, -4.0);

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

    std::cout << "[Step 5] Decrypting weights and running inference..." << std::endl;
    Plaintext ptxW;
    cc->Decrypt(keyPair.secretKey, w_enc, &ptxW);
    auto wVec = ptxW->GetRealPackedValue();

    std::vector<double> weights(f, 0.0);
    for (int j = 0; j < f; j++) {
        weights[j] = wVec[j];
    }

    std::cout << "\n  Weights: ";
    for (int j = 0; j < FH_RAW_FEATURES; j++) {
        std::cout << std::setw(8) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << "\n  Bias (w[" << FH_RAW_FEATURES << "]): " << weights[FH_RAW_FEATURES] << std::endl;

    expResult.classResult = performInference(weights, testSet, algorithmName);

    auto totalEnd = high_resolution_clock::now();
    expResult.totalTime = totalEnd - totalStart;
    expResult.valid = true;

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

std::vector<SFHCheckpointResult> runSimplifiedHessian(
                          std::shared_ptr<Encryption> enc,
                          CryptoContext<DCRTPoly> cc,
                          KeyPair<DCRTPoly> keyPair,
                          const std::vector<int>& rotIndices,
                          int multDepth,
                          const Ciphertext<DCRTPoly>& X_enc,
                          const Ciphertext<DCRTPoly>& y_enc,
                          const FHDataset& testSet,
                          const FHPrecomputed& pre,
                          const std::vector<int>& checkpoints,
                          bool verbose) {
    using namespace std::chrono;

    std::vector<SFHCheckpointResult> sfhResults;
    std::set<int> checkpointSet(checkpoints.begin(), checkpoints.end());
    int maxIter = checkpoints.empty() ? 256 : checkpoints.back();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  Simplified Fixed Hessian (Diagonal Inverse)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int f = FH_FEATURES;
    const int sampleDim = FH_MATRIX_DIM;
    const int batchSize = sampleDim * sampleDim;

    FH_AR24 algo(enc, cc, keyPair, rotIndices, multDepth, false);
    algo.setVerbose(verbose);

    auto totalStart = high_resolution_clock::now();
    std::cout << "\n[Step 1a] Computing X^T y..." << std::endl;
    auto xtyStart = high_resolution_clock::now();
    auto Xty_enc = algo.computeXty(X_enc, y_enc, f, sampleDim);
    auto xtyEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtyTime = xtyEnd - xtyStart;
    std::cout << "  X^T y time: " << std::fixed << std::setprecision(3) << xtyTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxXty;
        cc->Decrypt(keyPair.secretKey, Xty_enc, &ptxXty);
        auto xtyVec = ptxXty->GetRealPackedValue();
        std::cout << "  [X^T y] Encrypted (first 8): ";
        for (int i = 0; i < 8; i++) std::cout << std::setprecision(4) << xtyVec[i] << " ";
        std::cout << std::endl;
        double maxErr = 0;
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.Xty[0][i] - xtyVec[i]));
        std::cout << "  [X^T y] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }
    Xty_enc->SetSlots(f * f);
    std::cout << "[Step 1b] Computing X^T (transpose)..." << std::endl;
    auto xtStart = high_resolution_clock::now();
    auto Xt_enc = algo.eval_transpose(X_enc, sampleDim, batchSize);
    auto xtEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtTime = xtEnd - xtStart;
    std::cout << "  X^T time: " << std::fixed << std::setprecision(3) << xtTime.count() << " s" << std::endl;

    std::cout << "[Step 1c] Computing X^T X (JKLS18, " << sampleDim << "x" << sampleDim << ")..." << std::endl;
    auto xtxStart = high_resolution_clock::now();
    auto XtX_64 = algo.eval_mult_JKLS18(Xt_enc, X_enc, sampleDim);
    auto xtxEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtxTime = xtxEnd - xtxStart;
    std::cout << "  X^T X time: " << std::fixed << std::setprecision(3) << xtxTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxXtX64;
        cc->Decrypt(keyPair.secretKey, XtX_64, &ptxXtX64);
        auto xtx64Vec = ptxXtX64->GetRealPackedValue();
        std::cout << "  [JKLS18 raw] XtX_64[0,0] = " << xtx64Vec[0]
                  << " (expected: " << pre.XtX[0][0] << ")" << std::endl;
        std::cout << "  [JKLS18 raw] XtX_64[1,1] = " << xtx64Vec[sampleDim + 1]
                  << " (expected: " << pre.XtX[0][f + 1] << ")" << std::endl;
    }

    std::cout << "[Step 1d] Rebatching X^T X from " << sampleDim << "x" << sampleDim << " to " << f << "x" << f << "..." << std::endl;
    auto rbStart = high_resolution_clock::now();
    auto XtX_16 = algo.rebatchToFeatureSpace(XtX_64, sampleDim, f);
    auto rbEnd = high_resolution_clock::now();
    std::chrono::duration<double> rbTime = rbEnd - rbStart;
    std::cout << "  Rebatch time: " << std::fixed << std::setprecision(3) << rbTime.count() << " s" << std::endl;

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
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++)
            maxErr = std::max(maxErr, std::abs(pre.XtX[0][i * f + i] - xtxVec[i * f + i]));
        std::cout << "  [X^T X diag] Max error vs plaintext: " << std::scientific << maxErr << std::endl;
    }

    auto precompTime = xtyTime + xtTime + xtxTime + rbTime;
    std::cout << "  Precomputation subtotal: " << std::fixed << std::setprecision(3)
              << precompTime.count() << " s" << std::endl;
    std::cout << "\n[Step 2] Computing diagonal Hessian from X (encrypted)..." << std::endl;
    auto diagHStart = high_resolution_clock::now();

    auto diagH = algo.computeDiagHessian_enc(X_enc, sampleDim, f);

    auto diagHEnd = high_resolution_clock::now();
    std::chrono::duration<double> diagHTime = diagHEnd - diagHStart;
    std::cout << "  Diagonal Hessian time: " << std::fixed << std::setprecision(3)
              << diagHTime.count() << " s" << std::endl;

    {
        Plaintext ptxDiagH;
        cc->Decrypt(keyPair.secretKey, diagH, &ptxDiagH);
        auto diagHVec = ptxDiagH->GetRealPackedValue();

        std::cout << "\n  [diag(H̃)] Verification (first 14):" << std::endl;
        std::cout << "    Plaintext:  ";
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            double diagH_plain = (std::abs(pre.inv_diag_H[i * f + i]) > 1e-10)
                                 ? (1.0 / pre.inv_diag_H[i * f + i])
                                 : 0.0;
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << diagH_plain << " ";
        }
        std::cout << std::endl;
        std::cout << "    Encrypted:  ";
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << diagHVec[i] << " ";
        }
        std::cout << std::endl;

        double maxErr = 0;
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            double diagH_plain = (std::abs(pre.inv_diag_H[i * f + i]) > 1e-10)
                                 ? (1.0 / pre.inv_diag_H[i * f + i])
                                 : 0.0;
            maxErr = std::max(maxErr, std::abs(diagH_plain - diagHVec[i]));
        }
        std::cout << "    Max error:  " << std::scientific << maxErr << std::endl;
    }

    // ========== Step 3: Iterative diagonal inverse ==========
    std::cout << "[Step 3] Iterative diagonal inverse..." << std::endl;
    auto nrStart = high_resolution_clock::now();

    auto invDiag = algo.eval_diagonal_inverse(diagH, FH_BATCH_SIZE, f, sampleDim, 8);

    auto nrEnd = high_resolution_clock::now();
    std::chrono::duration<double> nrTime = nrEnd - nrStart;
    std::cout << "  Diagonal inverse time: " << std::fixed << std::setprecision(3)
              << nrTime.count() << " s" << std::endl;

    {
        Plaintext ptxInvDiag;
        cc->Decrypt(keyPair.secretKey, invDiag, &ptxInvDiag);
        auto invVec = ptxInvDiag->GetRealPackedValue();

        std::cout << "\n  [diag(H̃^{-1})] Verification (first 14):" << std::endl;
        std::cout << "    Plaintext:  ";
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << pre.inv_diag_H[i * f + i] << " ";
        }
        std::cout << std::endl;
        std::cout << "    Encrypted:  ";
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                      << invVec[i] << " ";
        }
        std::cout << std::endl;

        double maxErr = 0;
        for (int i = 0; i < FH_RAW_FEATURES + 1; i++) {
            maxErr = std::max(maxErr, std::abs(pre.inv_diag_H[i * f + i] - invVec[i]));
        }
        std::cout << "    Max error:  " << std::scientific << maxErr << std::endl;

        int actualFeatures = FH_RAW_FEATURES + 1;
        double u0 = -16.0 / (FH_BATCH_SIZE * actualFeatures);
        std::cout << "\n  [Diagonal Inverse Config]" << std::endl;
        std::cout << "    u0 = -16/(N*(d+1)) = " << u0 << std::endl;
        std::cout << "    N=" << FH_BATCH_SIZE << ", d+1=" << actualFeatures << std::endl;
    }

    int bootstrapCount = 0;
    const int slots16 = f * f;

    std::cout << "\n[Step 4] Iterative training (" << maxIter << " iterations)..." << std::endl;
    auto gradStart = high_resolution_clock::now();

    std::vector<double> w_vec(slots16, 0.0);
    for (int rep = 0; rep < f; rep++) {
        for (int j = 0; j <= FH_RAW_FEATURES; j++) {
            w_vec[rep * f + j] = 0.001;
        }
    }

    auto w_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(w_vec, 1, 0, nullptr, slots16));

    std::cout << "  Initial w level: " << w_enc->GetLevel() << "/" << multDepth << std::endl;

    Xty_enc->SetSlots(slots16);
    auto Xty_scaled = cc->EvalMult(Xty_enc, 0.5);

    auto XtX_16_dd = XtX_16->Clone();
    XtX_16_dd->SetSlots(slots16);

    auto bootIfNeeded = [&](Ciphertext<DCRTPoly>& ct) {
        if ((int)ct->GetLevel() >= multDepth - 1) {
            ct->SetSlots(f);
            ct = cc->EvalBootstrap(ct, 2, 18);
            ct->SetSlots(slots16);
            bootstrapCount++;
        }
    };

    for (int iter = 0; iter < maxIter; iter++) {
        auto w_col = algo.eval_transpose(w_enc, f, slots16);
        bootIfNeeded(w_col);

        auto XtXw = cc->EvalMultAndRelinearize(XtX_16_dd, w_col);
        for (int i = 0; i < (int)log2(f); i++) {
            int shift = f * (1 << i);
            cc->EvalAddInPlace(XtXw, cc->EvalRotate(XtXw, shift));
        }
        bootIfNeeded(XtXw);

        auto XtXw_scaled = cc->EvalMult(XtXw, 0.25);
        auto g = cc->EvalSub(Xty_scaled, XtXw_scaled);
        bootIfNeeded(g);

        auto delta_w = cc->EvalMultAndRelinearize(invDiag, g);

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

        w_enc = cc->EvalSub(w_enc, delta_w);

        std::cout << "  [Iter " << (iter + 1) << "] w level: " << w_enc->GetLevel()
                  << "/" << multDepth;

        if ((int)w_enc->GetLevel() >= multDepth - 1) {
            std::cout << " -> Bootstrapping (pre-level: " << w_enc->GetLevel() << "/" << multDepth << ")...";
            w_enc->SetSlots(f);
            w_enc = cc->EvalBootstrap(w_enc, 2, 18);
            w_enc->SetSlots(slots16);
            bootstrapCount++;
            std::cout << " done (new level: " << w_enc->GetLevel() << ")";
        }

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

            if (checkpointSet.count(iter + 1)) {
                auto checkpointTime = high_resolution_clock::now();
                std::chrono::duration<double> iterTime = checkpointTime - gradStart;
                double precompSec = (precompTime + diagHTime + nrTime).count();

                SFHCheckpointResult cpResult;
                cpResult.iterations = iter + 1;
                cpResult.timeSec = iterTime.count();
                cpResult.totalTimeSec = precompSec + iterTime.count();
                cpResult.classResult = performInference(weights, testSet,
                    "SFH (iter=" + std::to_string(iter + 1) + ")");
                sfhResults.push_back(cpResult);
            }
        }
        std::cout << std::endl;
    }

    auto gradEnd = high_resolution_clock::now();
    std::chrono::duration<double> gradTime = gradEnd - gradStart;

    std::cout << "  Total bootstraps: " << bootstrapCount << std::endl;

    std::cout << "\n[Step 5] Decrypting final weights..." << std::endl;

    Plaintext ptxW;
    cc->Decrypt(keyPair.secretKey, w_enc, &ptxW);
    auto wVec = ptxW->GetRealPackedValue();

    std::vector<double> weights(f, 0.0);
    for (int j = 0; j < f; j++) {
        weights[j] = wVec[j];
    }

    std::cout << "\n  Weights: ";
    for (int j = 0; j < FH_RAW_FEATURES; j++) {
        std::cout << std::setw(8) << std::setprecision(4) << std::fixed << weights[j] << " ";
    }
    std::cout << "\n  Bias (w[" << FH_RAW_FEATURES << "]): " << weights[FH_RAW_FEATURES] << std::endl;

    auto totalEnd = high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;

    std::cout << "\n--- Simplified Timing Summary ---" << std::endl;
    std::cout << "  X^T y:            " << std::setprecision(3) << xtyTime.count() << " s" << std::endl;
    std::cout << "  X^T:              " << xtTime.count() << " s" << std::endl;
    std::cout << "  X^T X:            " << xtxTime.count() << " s" << std::endl;
    std::cout << "  Rebatch:          " << rbTime.count() << " s" << std::endl;
    std::cout << "  Diagonal Hessian: " << diagHTime.count() << " s" << std::endl;
    std::cout << "  Diag inverse:     " << nrTime.count() << " s" << std::endl;
    std::cout << "  Gradient+update:  " << gradTime.count() << " s" << std::endl;
    std::cout << "  ---------------------------------" << std::endl;
    std::cout << "  TOTAL:            " << totalTime.count() << " s" << std::endl;

    return sfhResults;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
    #endif

    bool verbose = true;
    bool useBootstrapping = true;
    std::string algorithm = "both";  // "ar24", "newcol", "both", "simplified"
    int inversionIterations = 22;
    std::vector<int> sfhCheckpoints = {32, 64, 128, 256};

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
            sfhCheckpoints = {std::stoi(argv[++i])};
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
    std::cout << "#    FH (AR24/NewCol) + SFH Comprehensive Comparison           #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

#ifdef DATASET_DIABETES
    std::string trainPath = std::string(DATA_DIR) + "/diabetes_train.csv";
    std::string testPath = std::string(DATA_DIR) + "/diabetes_test.csv";
#else
    std::string trainPath = (FH_BATCH_SIZE == 64)
        ? std::string(DATA_DIR) + "/heart_train_64.csv"
        : std::string(DATA_DIR) + "/heart_train_128.csv";
    std::string testPath = std::string(DATA_DIR) + "/heart_test.csv";
#endif

    std::cout << "\n[1] Loading datasets..." << std::endl;
    auto trainSet = FHDataEncoder::loadCSV(trainPath);
    auto testSet = FHDataEncoder::loadCSV(testPath);

    FHDataEncoder::printDatasetInfo(trainSet, "Raw Train");
    FHDataEncoder::printDatasetInfo(testSet, "Raw Test");

    std::cout << "[2] Min-max normalization [0,1]..." << std::endl;
    FHDataEncoder::normalizeFeatures(trainSet);
    FHDataEncoder::normalizeWithParams(testSet, trainSet);

    std::cout << "[3] Adding bias column and padding to 16 features..." << std::endl;
    FHDataEncoder::addBiasAndPad(trainSet);
    FHDataEncoder::addBiasAndPad(testSet);

    FHDataEncoder::printDatasetInfo(trainSet, "Processed Train");
    FHDataEncoder::printDatasetInfo(testSet, "Processed Test");

    const int NUM_BATCHES = 1;
    auto pre = FHHETrainer::precompute(trainSet, NUM_BATCHES, verbose);

    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;

    int multDepth;
    uint32_t scalingModSize, firstModSize;

    multDepth = 31;
    scalingModSize = 59;
    firstModSize = 60;

    int sampleDim = FH_MATRIX_DIM;
    int batchSize = sampleDim * sampleDim;

    std::cout << "Sample dimension: " << sampleDim << "x" << sampleDim << std::endl;
    std::cout << "Feature dimension: " << FH_FEATURES << "x" << FH_FEATURES << std::endl;
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
        std::vector<uint32_t> levelBudget = {4, 5};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, FH_FEATURES * FH_FEATURES);
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, FH_FEATURES);
        std::cout << " Setup done. Generating keys..." << std::flush;
        cc->EvalBootstrapKeyGen(keyPair.secretKey, FH_FEATURES * FH_FEATURES);
        cc->EvalBootstrapKeyGen(keyPair.secretKey, FH_FEATURES);
        std::cout << " Done." << std::endl;
    }

    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);

    std::cout << "\n--- Encrypting Training Data ---" << std::endl;

    auto X_packed = FHDataEncoder::packBatchX(trainSet, 0);
    auto y_packed = FHDataEncoder::packBatchY(trainSet, 0);

    auto X_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(X_packed, 1, 0, nullptr, batchSize));
    auto y_enc = cc->Encrypt(keyPair.publicKey,
        cc->MakeCKKSPackedPlaintext(y_packed, 1, 0, nullptr, sampleDim));

    std::cout << "  X encrypted: " << batchSize << " slots" << std::endl;
    std::cout << "  y encrypted: " << sampleDim << " slots" << std::endl;

    FHExperimentResult ar24Result, newcolResult;
    std::vector<SFHCheckpointResult> sfhResults;

    if (algorithm == "ar24" || algorithm == "both") {
        ar24Result = runFixedHessian<FH_AR24>(
            "AR24", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            X_enc, y_enc, testSet, pre, inversionIterations, verbose);
    }

    if (algorithm == "newcol" || algorithm == "both") {
        newcolResult = runFixedHessian<FH_NewCol>(
            "NewCol", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            X_enc, y_enc, testSet, pre, inversionIterations, verbose);
    }

    if (algorithm == "simplified" || algorithm == "both") {
        sfhResults = runSimplifiedHessian(enc, cc, keyPair, rotIndices, multDepth,
                             X_enc, y_enc, testSet, pre, sfhCheckpoints, verbose);
    }

    if (algorithm == "both" && ar24Result.valid && newcolResult.valid) {
        std::cout << "\n";
        std::cout << "================================================================" << std::endl;
        std::cout << "  Fixed Hessian - Comprehensive Comparison (Diabetes)" << std::endl;
        std::cout << "  Train: " << FH_BATCH_SIZE << " samples, Test: "
                  << testSet.numSamples << " samples, Features: " << FH_FEATURES << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "  " << std::left << std::setw(20) << "Method"
                  << " | " << std::setw(4) << "Iter"
                  << " | " << std::setw(9) << "Time (s)"
                  << " | " << std::setw(8) << "Accuracy"
                  << " | " << std::setw(8) << "F1 Score" << std::endl;
        std::cout << "  " << std::string(60, '-') << std::endl;

        std::cout << "  " << std::left << std::setw(20) << "FH (AR24)"
                  << " | " << std::right << std::setw(4) << 1
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2)
                  << ar24Result.totalTime.count()
                  << " | " << std::setw(7) << std::setprecision(2)
                  << ar24Result.classResult.accuracy() << "%"
                  << " | " << std::setw(7) << std::setprecision(2)
                  << ar24Result.classResult.f1Score() << "%" << std::endl;

        std::cout << "  " << std::left << std::setw(20) << "FH (NewCol)"
                  << " | " << std::right << std::setw(4) << 1
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2)
                  << newcolResult.totalTime.count()
                  << " | " << std::setw(7) << std::setprecision(2)
                  << newcolResult.classResult.accuracy() << "%"
                  << " | " << std::setw(7) << std::setprecision(2)
                  << newcolResult.classResult.f1Score() << "%" << std::endl;

        for (const auto& cp : sfhResults) {
            std::cout << "  " << std::left << std::setw(20) << "SFH"
                      << " | " << std::right << std::setw(4) << cp.iterations
                      << " | " << std::setw(9) << std::fixed << std::setprecision(2)
                      << cp.totalTimeSec
                      << " | " << std::setw(7) << std::setprecision(2)
                      << cp.classResult.accuracy() << "%"
                      << " | " << std::setw(7) << std::setprecision(2)
                      << cp.classResult.f1Score() << "%" << std::endl;
        }

        std::cout << "================================================================" << std::endl;
    }

    saveFHResults("fh_results.txt", ar24Result, newcolResult, sfhResults, (int)testSet.numSamples);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
