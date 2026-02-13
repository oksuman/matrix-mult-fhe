// main_encrypted.cpp
// Encrypted Logistic Regression (Fixed Hessian) with AR24 vs NewCol comparison
// Heart Disease dataset, CKKS homomorphic encryption

#include "lr_data_encoder.h"
#include "lr_he_trainer.h"
#include "lr_encrypted_ar24.h"
#include "lr_encrypted_newcol.h"
#include "encryption.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>

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
void performInference(const std::vector<double>& weights,
                      const LRDataset& testSet,
                      const std::string& methodName) {
    int correct = 0;
    int total = testSet.numSamples;
    int tp = 0, fp = 0, fn = 0, tn = 0;

    for (size_t i = 0; i < testSet.numSamples; i++) {
        double z = 0.0;
        for (int j = 0; j < LR_FEATURES; j++) {
            z += testSet.samples[i][j] * weights[j];
        }

        double pred_prob = 0.5 + z / 4.0;
        int pred_label = (pred_prob >= 0.5) ? 1 : -1;
        int true_label = testSet.labels[i];

        if (pred_label == true_label) correct++;
        if (pred_label == 1 && true_label == 1) tp++;
        else if (pred_label == 1 && true_label == -1) fp++;
        else if (pred_label == -1 && true_label == 1) fn++;
        else tn++;
    }

    double accuracy = 100.0 * correct / total;
    double precision = (tp + fp > 0) ? 100.0 * tp / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? 100.0 * tp / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "\n  [" << methodName << "] Inference Results:" << std::endl;
    std::cout << "    Correct: " << correct << " / " << total << std::endl;
    std::cout << "    Accuracy:  " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "    Precision: " << precision << "%" << std::endl;
    std::cout << "    Recall:    " << recall << "%" << std::endl;
    std::cout << "    F1 Score:  " << f1 << "%" << std::endl;
    std::cout << "    (TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn << ")" << std::endl;
}

// Run Fixed Hessian training with a given inversion algorithm
template<typename InvAlgorithm>
void runFixedHessian(const std::string& algorithmName,
                     std::shared_ptr<Encryption> enc,
                     CryptoContext<DCRTPoly> cc,
                     KeyPair<DCRTPoly> keyPair,
                     const std::vector<int>& rotIndices,
                     int multDepth,
                     bool useBootstrapping,
                     const Ciphertext<DCRTPoly>& XtX_16,     // 16x16 (rebatched, s-replicated)
                     const Ciphertext<DCRTPoly>& Xty_16,     // 16-dim vector in 16x16 slots
                     const LRDataset& testSet,
                     int inversionIterations,
                     bool verbose) {
    using namespace std::chrono;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  Fixed Hessian with " << algorithmName << " Matrix Inversion" << std::endl;
    std::cout << "  Bootstrapping: " << (useBootstrapping ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    InvAlgorithm algo(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);
    algo.setVerbose(verbose);

    const int f = LR_FEATURES;  // 16

    // Step 1: Invert XtX (which is PSD, directly invertible)
    // H^{-1} = -4 * (XtX)^{-1}
    std::cout << "\n[Step 1] Computing (X^T X)^{-1} with " << algorithmName << "..." << std::endl;
    auto invStart = high_resolution_clock::now();

    // XtX is in rebatched form (d*d*s slots from rebatchToFeatureSpace)
    // Need to get it back to d*d slots for inversion input
    auto XtX_for_inv = XtX_16->Clone();
    XtX_for_inv->SetSlots(f * f);

    // trace(XtX) upper bound: for min-max [0,1] normalized data with 64 samples
    // each feature in [0,1], XtX[i][i] <= 64, trace <= 64*14 ~ 896
    double traceUpperBound = 64.0 * 16;
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

    // Step 2: Compute gradient g = 0.5*Xty - 0.25*XtX*w_0
    std::cout << "[Step 2] Computing gradient..." << std::endl;
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
    auto Xty_scaled = cc->EvalMult(Xty_16, 0.5);
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

    // Step 3: delta_w = -4 * XtX_inv * g
    // w = w_0 - H^{-1} * g = w_0 - (-4 * XtX_inv * g) = w_0 + 4 * XtX_inv * g
    std::cout << "[Step 3] Computing weight update..." << std::endl;

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

    // Step 4: Decrypt and infer
    std::cout << "[Step 4] Decrypting weights and running inference..." << std::endl;
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
    std::cout << "\n  Bias (w[13]): " << weights[LR_RAW_FEATURES] << std::endl;

    performInference(weights, testSet, algorithmName + " Fixed Hessian");

    std::cout << "\n--- " << algorithmName << " Timing Summary ---" << std::endl;
    std::cout << "  Inversion:      " << std::setprecision(3) << invTime.count() << " s" << std::endl;
    std::cout << "  Gradient+update: " << gradTime.count() << " s" << std::endl;
    std::cout << "  Total:          " << (invTime + gradTime).count() << " s" << std::endl;
}

// Run Simplified Fixed Hessian (diagonal inverse, plaintext precomputed)
void runSimplifiedHessian(CryptoContext<DCRTPoly> cc,
                          KeyPair<DCRTPoly> keyPair,
                          const std::vector<int>& rotIndices,
                          const Ciphertext<DCRTPoly>& XtX_16,
                          const Ciphertext<DCRTPoly>& Xty_16,
                          const LRPrecomputed& pre,
                          const LRDataset& testSet,
                          bool verbose) {
    using namespace std::chrono;
    RotationComposer rot_local(cc, rotIndices, cc->GetRingDimension() / 2);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  Simplified Fixed Hessian (Diagonal Inverse)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    const int f = LR_FEATURES;  // 16
    auto simplStart = high_resolution_clock::now();

    // Initial weights
    std::vector<double> w0(f, 0.0);
    for (int j = 0; j <= LR_RAW_FEATURES; j++) {
        w0[j] = 0.001;
    }

    // Gradient: g = 0.5*Xty - 0.25*XtX*w_0
    // XtX * w_0 using plaintext
    std::vector<double> w0_T(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w0_T[i * f + j] = w0[i];
        }
    }
    auto w0_T_ptx = cc->MakeCKKSPackedPlaintext(w0_T, 1, 0, nullptr, f * f);

    auto XtX_16_dd = XtX_16->Clone();
    XtX_16_dd->SetSlots(f * f);
    auto XtXw = cc->EvalMult(XtX_16_dd, w0_T_ptx);

    // Row folding sum
    for (int i = 0; i < (int)log2(f); i++) {
        int shift = f * (1 << i);
        cc->EvalAddInPlace(XtXw, rot_local.rotate(XtXw, shift));
    }

    auto Xty_scaled = cc->EvalMult(Xty_16, 0.5);
    auto XtXw_scaled = cc->EvalMult(XtXw, 0.25);
    auto g = cc->EvalSub(Xty_scaled, XtXw_scaled);

    // delta_w = inv_diag_H * g (plaintext diagonal inverse * encrypted gradient)
    // inv_diag_H is precomputed plaintext 16x16 diagonal matrix
    // For mat-vec: inv_diag_H[i][j] * g_T[i][j], where g_T is column-replicated

    // Build column-replicated g: transpose
    auto g_rep = g->Clone();
    g_rep->SetSlots(f * f);

    // For diagonal matrix, we can just do element-wise multiply
    // inv_diag_H is diagonal: row i has inv_diag_H[i][i] at position (i,i), 0 elsewhere
    // Hadamard(inv_diag_H, g_T): only diagonal elements survive if g is column-replicated
    // Simpler: use plaintext inv_diag_H as row-replicated mask
    // Build plaintext: position(i,j) = inv_diag_H[j][j] for all i
    std::vector<double> inv_diag_rep(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            inv_diag_rep[i * f + j] = pre.inv_diag_H[j * f + j];
        }
    }
    auto inv_diag_ptx = cc->MakeCKKSPackedPlaintext(inv_diag_rep, 1, 0, nullptr, f * f);

    // delta_w[j] = inv_diag_H[j][j] * g[j]
    // g is in row 0 (row-replicated after folding), position(0,j) = g[j]
    auto delta_w = cc->EvalMult(g_rep, inv_diag_ptx);

    // w = w_0 - delta_w
    std::vector<double> w0_rep(f * f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w0_rep[i * f + j] = w0[j];
        }
    }
    auto w0_ptx = cc->MakeCKKSPackedPlaintext(w0_rep, 1, 0, nullptr, f * f);
    auto w_enc = cc->EvalSub(w0_ptx, delta_w);

    auto simplEnd = high_resolution_clock::now();
    std::chrono::duration<double> simplTime = simplEnd - simplStart;

    // Decrypt and infer
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
    std::cout << "\n  Bias (w[13]): " << weights[LR_RAW_FEATURES] << std::endl;

    performInference(weights, testSet, "Simplified Fixed Hessian");

    std::cout << "\n--- Simplified Timing ---" << std::endl;
    std::cout << "  Total: " << std::setprecision(3) << simplTime.count() << " s" << std::endl;
}

int main(int argc, char* argv[]) {
    bool verbose = true;
    bool useBootstrapping = true;
    std::string algorithm = "both";  // "ar24", "newcol", "both"
    int inversionIterations = 25;

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
        } else if (arg == "--iterations" && i + 1 < argc) {
            inversionIterations = std::stoi(argv[++i]);
        }
    }

    std::cout << "\n";
    std::cout << "###############################################################" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "#    Logistic Regression (Fixed Hessian) - Encrypted Mode     #" << std::endl;
    std::cout << "#    Heart Disease Dataset                                    #" << std::endl;
    std::cout << "#    AR24 vs NewCol Matrix Inversion Comparison               #" << std::endl;
    std::cout << "#                                                             #" << std::endl;
    std::cout << "###############################################################" << std::endl;

    // ========== Step 1: Load and preprocess data ==========
    std::string trainPath = std::string(DATA_DIR) + "/heart_train.csv";
    std::string testPath = std::string(DATA_DIR) + "/heart_test.csv";

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

    if (useBootstrapping) {
        multDepth = 29;
        scalingModSize = 59;
        firstModSize = 60;
    } else {
        multDepth = 30;
        scalingModSize = 50;
        firstModSize = 50;
    }

    int sampleDim = LR_MATRIX_DIM;  // 64
    int batchSize = sampleDim * sampleDim;  // 4096

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
        std::vector<uint32_t> levelBudget = {4, 5};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, LR_FEATURES * LR_FEATURES);
        std::cout << " Setup done. Generating keys..." << std::flush;
        cc->EvalBootstrapKeyGen(keyPair.secretKey, LR_FEATURES * LR_FEATURES);
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

    // ========== Step 4: Common precomputation (encrypted) ==========
    // Create a temporary base class instance for common operations
    LR_AR24 baseOps(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping);
    baseOps.setVerbose(verbose);

    using namespace std::chrono;

    // 4a. X^T y
    std::cout << "\n[4a] Computing X^T y (encrypted)..." << std::endl;
    auto xtyStart = high_resolution_clock::now();
    auto Xty_enc = baseOps.computeXty(X_enc, y_enc, LR_FEATURES, sampleDim);
    auto xtyEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtyTime = xtyEnd - xtyStart;
    std::cout << "  X^T y time: " << std::fixed << std::setprecision(3) << xtyTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxXty;
        cc->Decrypt(keyPair.secretKey, Xty_enc, &ptxXty);
        auto xtyVec = ptxXty->GetRealPackedValue();
        std::cout << "  X^T y (first 8): ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::setprecision(4) << xtyVec[i] << " ";
        }
        std::cout << std::endl;
    }

    // Set Xty to 16x16 format (replicate to 256 slots)
    Xty_enc->SetSlots(LR_FEATURES * LR_FEATURES);

    // 4b. X^T (transpose)
    std::cout << "[4b] Computing X^T (transpose)..." << std::endl;
    auto xtStart = high_resolution_clock::now();
    auto Xt_enc = baseOps.eval_transpose(X_enc, sampleDim, batchSize);
    auto xtEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtTime = xtEnd - xtStart;
    std::cout << "  X^T time: " << std::fixed << std::setprecision(3) << xtTime.count() << " s" << std::endl;

    // 4c. X^T X (JKLS18 in 64x64 space)
    std::cout << "[4c] Computing X^T X (JKLS18, 64x64)..." << std::endl;
    auto xtxStart = high_resolution_clock::now();
    auto XtX_64 = baseOps.eval_mult_JKLS18(Xt_enc, X_enc, sampleDim);
    auto xtxEnd = high_resolution_clock::now();
    std::chrono::duration<double> xtxTime = xtxEnd - xtxStart;
    std::cout << "  X^T X time: " << std::fixed << std::setprecision(3) << xtxTime.count() << " s" << std::endl;

    if (verbose) {
        Plaintext ptxXtX;
        cc->Decrypt(keyPair.secretKey, XtX_64, &ptxXtX);
        auto xtxVec = ptxXtX->GetRealPackedValue();
        std::cout << "  X^T X diagonal (64x64, first 8): ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::setprecision(4) << xtxVec[i * sampleDim + i] << " ";
        }
        std::cout << std::endl;
    }

    // 4d. Rebatch 64x64 -> 16x16
    std::cout << "[4d] Rebatching X^T X from 64x64 to 16x16..." << std::endl;
    auto rbStart = high_resolution_clock::now();
    auto XtX_16 = baseOps.rebatchToFeatureSpace(XtX_64, sampleDim, LR_FEATURES);
    auto rbEnd = high_resolution_clock::now();
    std::chrono::duration<double> rbTime = rbEnd - rbStart;
    std::cout << "  Rebatch time: " << std::fixed << std::setprecision(3) << rbTime.count() << " s" << std::endl;

    if (verbose) {
        auto XtX_16_tmp = XtX_16->Clone();
        XtX_16_tmp->SetSlots(LR_FEATURES * LR_FEATURES);
        Plaintext ptxRb;
        cc->Decrypt(keyPair.secretKey, XtX_16_tmp, &ptxRb);
        auto rbVec = ptxRb->GetRealPackedValue();
        std::cout << "  X^T X (16x16) diagonal: ";
        for (int i = 0; i < 8; i++) {
            std::cout << std::setprecision(4) << rbVec[i * LR_FEATURES + i] << " ";
        }
        std::cout << std::endl;
        double tr = 0;
        for (int i = 0; i < LR_FEATURES; i++) tr += rbVec[i * LR_FEATURES + i];
        std::cout << "  trace(X^T X) = " << std::setprecision(4) << tr << std::endl;
    }

    std::cout << "\n--- Precomputation Summary ---" << std::endl;
    std::cout << "  X^T y:    " << std::setprecision(3) << xtyTime.count() << " s" << std::endl;
    std::cout << "  X^T:      " << xtTime.count() << " s" << std::endl;
    std::cout << "  X^T X:    " << xtxTime.count() << " s" << std::endl;
    std::cout << "  Rebatch:  " << rbTime.count() << " s" << std::endl;
    std::cout << "  Total:    " << (xtyTime + xtTime + xtxTime + rbTime).count() << " s" << std::endl;

    // ========== Step 5: Run Fixed Hessian with AR24 ==========
    if (algorithm == "ar24" || algorithm == "both") {
        runFixedHessian<LR_AR24>(
            "AR24", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            XtX_16, Xty_enc, testSet, inversionIterations, verbose);
    }

    // ========== Step 6: Run Fixed Hessian with NewCol ==========
    if (algorithm == "newcol" || algorithm == "both") {
        runFixedHessian<LR_NewCol>(
            "NewCol", enc, cc, keyPair, rotIndices, multDepth, useBootstrapping,
            XtX_16, Xty_enc, testSet, inversionIterations, verbose);
    }

    // ========== Step 7: Run Simplified Fixed Hessian ==========
    runSimplifiedHessian(cc, keyPair, rotIndices, XtX_16, Xty_enc,
                         pre, testSet, verbose);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  All experiments completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
