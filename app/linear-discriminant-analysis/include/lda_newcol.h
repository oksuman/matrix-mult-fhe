// LDA implementation using NewCol algorithm for matrix inversion
#pragma once

#include "lda_encrypted_base.h"

class LDA_NewCol : public LDAEncryptedBase {
private:
    // NewCol-specific mask generation
    std::vector<double> generateMaskVector(int batch_size, int k, int d) {
        std::vector<double> result(batch_size, 0.0);
        for (int i = k * d * d; i < (k + 1) * d * d; ++i) {
            result[i] = 1.0;
        }
        return result;
    }

    std::vector<double> genDiagVector(int k, int diag_index, int d) {
        std::vector<double> result(d * d, 0.0);

        if (diag_index < 1 || diag_index > d * d ||
            (diag_index > d && diag_index < d * d - (d - 1))) {
            return result;
        }

        for (int i = 0; i < d; ++i) {
            result[i * d + ((i + k) % d)] = 1.0;
        }

        int rotation = 0;
        bool right_rotation = false;

        if (diag_index <= d) {
            rotation = diag_index - 1;
        } else {
            right_rotation = true;
            rotation = d * d - diag_index + 1;
        }

        if (rotation > 0) {
            for (int i = 0; i < rotation; ++i) {
                for (int j = 0; j < d; ++j) {
                    if (right_rotation) {
                        result[j * d + (d - 1 - i)] = 0.0;
                    } else {
                        result[j * d + i] = 0.0;
                    }
                }
            }
        }

        std::vector<double> rotated(d * d, 0.0);
        for (int i = 0; i < d * d; ++i) {
            int new_pos;
            if (right_rotation) {
                new_pos = (i + rotation) % d + (i / d) * d;
            } else {
                new_pos = (i + d - rotation) % d + (i / d) * d;
            }
            rotated[new_pos] = result[i];
        }

        return rotated;
    }

    std::vector<double> genBatchDiagVector(int s, int k, int diag_index, int d) {
        std::vector<double> result;
        result.reserve(d * d * s);

        for (int i = 0; i < s; ++i) {
            std::vector<double> diag_vector = genDiagVector(k + i, diag_index, d);
            result.insert(result.end(), diag_vector.begin(), diag_vector.end());
        }

        return result;
    }

    Ciphertext<DCRTPoly> vecRotsOpt(const std::vector<Ciphertext<DCRTPoly>>& matrixM,
                                    int is, int s, int np, int d) {
        auto rotsM = getZeroCiphertext(d * d * s);

        for (int j = 0; j < s / np; j++) {
            auto T = getZeroCiphertext(d * d * s);

            for (int i = 0; i < np; i++) {
                auto msk = generateMaskVector(d * d * s, np * j + i, d);
                msk = vectorRotate(msk, -is * d * s - j * d * np);

                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d * s);
                m_cc->EvalAddInPlace(T, m_cc->EvalMult(matrixM[i], pmsk));
            }
            m_cc->EvalAddInPlace(rotsM, rot.rotate(T, is * d * s + j * d * np));
        }

        return rotsM;
    }

    // NewCol matrix multiplication
    Ciphertext<DCRTPoly> eval_mult_NewCol(const Ciphertext<DCRTPoly>& matrixA,
                                          const Ciphertext<DCRTPoly>& matrixB,
                                          int s, int B, int ng, int nb, int np, int d) {
        auto matrixC = getZeroCiphertext(d * d * s);
        Ciphertext<DCRTPoly> babyStepsOfA[nb];
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfB;

        // Baby steps for A: nb rotations
        for (int i = 0; i < nb; i++) {
            babyStepsOfA[i] = rot.rotate(matrixA, i);
        }

        // Baby steps for B: np rotations
        for (int i = 0; i < np; i++) {
            auto t = rot.rotate(matrixB, i * d);
            t->SetSlots(d * d * s);
            babyStepsOfB.push_back(t);
        }

        for (int i = 0; i < B; i++) {
            auto batched_rotations_B = vecRotsOpt(babyStepsOfB, i, s, np, d);
            auto diagA = getZeroCiphertext(d * d * s);

            for (int k = -ng; k < ng; k++) {
                if (k < 0) {
                    auto tmp = getZeroCiphertext(d * d * s);
                    auto babyStep = (k == -ng) ? 1 : 0;

                    for (int j = d * d + k * nb + 1 + babyStep; j <= d * d + (k + 1) * nb; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j, d), -k * nb);
                        m_cc->EvalAddInPlace(tmp,
                            m_cc->EvalMult(babyStepsOfA[babyStep],
                                m_cc->MakeCKKSPackedPlaintext(rotated_plain_vec, 1, 0, nullptr, s * d * d)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb));
                } else {
                    auto tmp = getZeroCiphertext(d * d * s);
                    auto babyStep = 0;

                    for (int j = k * nb + 1; j <= (k + 1) * nb; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j, d), -k * nb);
                        m_cc->EvalAddInPlace(tmp,
                            m_cc->EvalMult(babyStepsOfA[babyStep],
                                m_cc->MakeCKKSPackedPlaintext(rotated_plain_vec, 1, 0, nullptr, d * d * s)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb));
                }
            }

            m_cc->EvalAddInPlace(matrixC, m_cc->EvalMult(diagA, batched_rotations_B));
        }

        // Final accumulation
        for (int i = 1; i <= (int)log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC, rot.rotate(matrixC, (d * d * s) / (1 << i)));
        }
        matrixC->SetSlots(d * d);

        return matrixC;
    }

public:
    LDA_NewCol(std::shared_ptr<Encryption> enc,
               CryptoContext<DCRTPoly> cc,
               KeyPair<DCRTPoly> keyPair,
               std::vector<int> rotIndices,
               int multDepth,
               bool useBootstrapping = true)
        : LDAEncryptedBase(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping)
    {}

    bool m_verbose = false;

    // Scalar inverse using power series: computes 1/t given encrypted t and upper_bound u >= t
    // Algorithm: x = 1/u, t_bar = 1 - t/u, iterate x = x*(1+t_bar), t_bar = t_bar^2
    // batchSize: number of slots to replicate the result
    Ciphertext<DCRTPoly> eval_scalar_inverse(const Ciphertext<DCRTPoly>& t, double upperBound, int iterations, int batchSize) {
        // x_0 = 1/upperBound replicated to all slots
        double x0 = 1.0 / upperBound;
        auto x = m_cc->Encrypt(m_keyPair.publicKey,
            m_cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, x0), 1, 0, nullptr, batchSize));

        // t_bar = 1 - t/upperBound
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        if (m_verbose) {
            std::cout << "  [Scalar Inv] upper_bound = " << upperBound << ", x0 = " << x0 << std::endl;
        }

        for (int i = 0; i < iterations; i++) {
            // x = x * (1 + t_bar)
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            // t_bar = t_bar * t_bar
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }

        return x;
    }

    // Matrix inversion with actualDim specifying actual matrix size (rest is padding)
    // traceUpperBound = N * d for Z-score normalization
    Ciphertext<DCRTPoly> eval_inverse_impl(const Ciphertext<DCRTPoly>& M, int d, int iterations, int actualDim, double traceUpperBound = 0) {
        int maxBatch = m_cc->GetRingDimension() / 2;
        int s = std::min(d, maxBatch / d / d);
        s = std::max(1, s);

        int B = d / s;
        int ng = 2;
        int nb = 4;
        int np = 2;

        if (d >= 16) {
            ng = 4;
            nb = 4;
            np = 4;
        }

        // Identity matrix: 1s only at actual feature positions (0..actualDim-1), 0s at padding
        std::vector<double> vI(d * d, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI[i * d + i] = 1.0;
        }
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);
        auto I_enc = m_cc->Encrypt(m_keyPair.publicKey, pI);

        if (m_verbose && actualDim < d) {
            std::cout << "  [Inversion] Identity: 1s at 0.." << actualDim-1 << ", 0s at " << actualDim << ".." << d-1 << std::endl;
        }

        // Compute trace(M) in encrypted form
        auto traceEnc = eval_trace(M, d, d * d);

        // Compute 1/trace using scalar power series (NO decryption)
        // Use traceUpperBound as the upper bound for scalar inverse
        // For Z-score normalized data: trace(S_W) <= N * d (samples * features)
        if (traceUpperBound <= 0) {
            // Fallback: N * d where N ~ 64 samples typical
            traceUpperBound = 64.0 * actualDim;
        }

        // Compute encrypted alpha = 1/trace using power series (3 iterations for scalar)
        auto alphaEnc = eval_scalar_inverse(traceEnc, traceUpperBound, 3, d * d);

        if (m_verbose) {
            // For debugging: decrypt and show actual trace and alpha
            Plaintext ptxTrace, ptxAlpha;
            m_cc->Decrypt(m_keyPair.secretKey, traceEnc, &ptxTrace);
            m_cc->Decrypt(m_keyPair.secretKey, alphaEnc, &ptxAlpha);
            std::cout << "  [Inversion] trace(M) = " << ptxTrace->GetRealPackedValue()[0]
                      << ", alpha = " << ptxAlpha->GetRealPackedValue()[0] << std::endl;
        }

        // Y_0 = alpha * I (encrypted), with I having zeros at padding positions
        // Since alphaEnc is scalar replicated, EvalMult broadcasts it
        auto Y = m_cc->EvalMult(I_enc, alphaEnc);

        // A_bar = I - alpha * M (I has zeros at padding, so A_bar also has zeros there)
        auto A_bar = m_cc->EvalSub(I_enc, m_cc->EvalMult(M, alphaEnc));

        if (m_verbose) {
            std::cout << "  [Inversion Init] Y level: " << Y->GetLevel()
                      << ", A_bar level: " << A_bar->GetLevel() << std::endl;
        }

        for (int i = 0; i < iterations - 1; i++) {
            if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
                if (m_verbose) {
                    std::cout << "  [Iter " << i << "] Bootstrapping triggered. Y level: "
                              << Y->GetLevel() << std::endl;
                }
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
                if (m_verbose) {
                    std::cout << "           After bootstrap. Y level: " << Y->GetLevel()
                              << ", A_bar level: " << A_bar->GetLevel() << std::endl;
                }
            }

            Y = eval_mult_NewCol(Y, m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
            A_bar = eval_mult_NewCol(A_bar, A_bar, s, B, ng, nb, np, d);

            if (m_verbose && (i % 5 == 0 || i == iterations - 2)) {
                std::cout << "  [Iter " << i << "] Y level: " << Y->GetLevel()
                          << ", A_bar level: " << A_bar->GetLevel() << std::endl;
            }
        }

        // Bootstrap before final multiplication if needed
        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
            if (m_verbose) {
                std::cout << "  [Before Final] Bootstrapping. Y level: " << Y->GetLevel() << std::endl;
            }
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
            if (m_verbose) {
                std::cout << "           After bootstrap. Y level: " << Y->GetLevel() << std::endl;
            }
        }

        Y = eval_mult_NewCol(Y, m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);

        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
            if (m_verbose) {
                std::cout << "  [Final] Bootstrapping. Y level before: " << Y->GetLevel() << std::endl;
            }
            Y = m_cc->EvalBootstrap(Y, 2, 18);
            if (m_verbose) {
                std::cout << "  [Final] Y level after: " << Y->GetLevel() << std::endl;
            }
        }

        if (m_verbose) {
            std::cout << "  [Inversion Done] Final Y level: " << Y->GetLevel() << std::endl;
        }

        return Y;
    }

    // Default eval_inverse uses full identity (for base class compatibility)
    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d, int iterations) override {
        return eval_inverse_impl(M, d, iterations, d);
    }

    // Binary LDA training: w = S_W^{-1} * (μ_1 - μ_0)
    LDAEncryptedResult trainWithTimings(
        const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
        const LDADataset& dataset,
        int inversionIterations,
        LDATimingResult& timings,
        bool verbose = false,
        bool /*sbOnly*/ = false) override
    {
        using namespace std::chrono;
        auto totalStart = high_resolution_clock::now();

        LDAEncryptedResult result;
        size_t f = dataset.numFeatures;
        size_t f_tilde = dataset.paddedFeatures;
        size_t numClasses = dataset.numClasses;
        int largeDim = std::max(dataset.paddedSamples, dataset.paddedFeatures);

        result.classCounts = dataset.samplesPerClass;
        result.classMeans.resize(numClasses);

        m_verbose = verbose;

        if (verbose) {
            std::cout << "\n========== LDA Training (NewCol Encrypted) ==========" << std::endl;
            std::cout << "Features: " << f << " (padded: " << f_tilde << ")" << std::endl;
            std::cout << "Samples: " << dataset.numSamples << std::endl;
            std::cout << "Matrix dimension: " << largeDim << "x" << largeDim << std::endl;
            std::cout << "Bootstrapping: " << (m_useBootstrapping ? "enabled" : "disabled") << std::endl;
            std::cout << "Multiplicative depth: " << m_multDepth << std::endl;
        }

        // ========== Step 1: Compute Class Means ==========
        if (verbose) std::cout << "\n[Step 1] Computing class means..." << std::endl;
        auto meanStart = high_resolution_clock::now();

        result.classMeansEncrypted.resize(numClasses);
        std::vector<Ciphertext<DCRTPoly>> classMeanForSw(numClasses);
        std::vector<Ciphertext<DCRTPoly>> classMeanVec(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];

            // Mean for S_W: masked division (zeros in padding rows)
            classMeanForSw[c] = eval_computeMeanForSw(classDataEncrypted[c], s_c, f, largeDim);

            // Mean vector: scalar division, extract to f_tilde slots
            classMeanVec[c] = eval_computeMeanForSb(classDataEncrypted[c], s_c, f_tilde, largeDim);
            result.classMeansEncrypted[c] = classMeanVec[c]->Clone();

            // Decrypt for plaintext inference
            Plaintext ptx;
            m_cc->Decrypt(m_keyPair.secretKey, classMeanVec[c], &ptx);
            std::vector<double> meanVec = ptx->GetRealPackedValue();
            result.classMeans[c] = std::vector<double>(meanVec.begin(), meanVec.begin() + f);

            if (verbose) {
                debugPrintLevel("Class " + std::to_string(c) + " mean (for S_W)", classMeanForSw[c]);
                debugPrintVector("Class " + std::to_string(c) + " Mean", classMeanVec[c], f);

                std::cout << "  Mean replication check for S_W (rows 0,1,2):" << std::endl;
                Plaintext ptxSw;
                m_cc->Decrypt(m_keyPair.secretKey, classMeanForSw[c], &ptxSw);
                std::vector<double> meanSwVals = ptxSw->GetRealPackedValue();
                for (int row = 0; row < 3 && row < (int)s_c; row++) {
                    std::cout << "    Row " << row << ": ";
                    for (size_t j = 0; j < std::min(f, (size_t)5); j++) {
                        std::cout << std::setprecision(4) << std::fixed << meanSwVals[row * largeDim + j] << " ";
                    }
                    std::cout << "..." << std::endl;
                }
                std::cout << std::flush;
            }
        }

        // Compute global mean from encrypted class means (weighted average)
        // μ = (n_0 * μ_0 + n_1 * μ_1 + ...) / (n_0 + n_1 + ...)
        // Both classMeanVec[c] and globalMeanEnc are in 16-replicated form
        auto globalMeanEnc = eval_computeGlobalMean(classMeanVec, dataset.samplesPerClass, f_tilde);
        result.globalMeanEncrypted = globalMeanEnc->Clone();

        // Decrypt global mean for debugging/inference only
        {
            Plaintext ptxGlobal;
            m_cc->Decrypt(m_keyPair.secretKey, globalMeanEnc, &ptxGlobal);
            std::vector<double> globalMeanVec = ptxGlobal->GetRealPackedValue();
            result.globalMean = std::vector<double>(globalMeanVec.begin(), globalMeanVec.begin() + f);
        }

        if (verbose) {
            std::cout << "=== Global Mean (from class means, len=" << f << ") ===" << std::endl;
            for (size_t i = 0; i < f; i++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed << result.globalMean[i] << " ";
            }
            std::cout << std::endl << std::endl << std::flush;
        }

        auto meanEnd = high_resolution_clock::now();
        timings.meanComputation = meanEnd - meanStart;

        // ========== Step 2: Compute S_W (Within-class scatter) ==========
        if (verbose) std::cout << "[Step 2] Computing S_W (within-class scatter)..." << std::endl;
        auto swStart = high_resolution_clock::now();

        auto Sw = getZeroCiphertext(largeDim * largeDim);
        result.X_bar_c_decrypted.resize(numClasses);
        result.S_c_decrypted.resize(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];

            if (verbose) {
                std::cout << "  Class " << c << " (n=" << s_c << "):" << std::endl;
                std::cout << "    Computing X_bar = X - mu..." << std::endl;
            }

            // X_bar_c = X_c - μ_c (both are 256×256, μ has zeros in padding rows)
            auto X_bar_c = m_cc->EvalSub(classDataEncrypted[c], classMeanForSw[c]);

            // Decrypt and store X_bar_c for debugging
            {
                Plaintext ptxXbar;
                m_cc->Decrypt(m_keyPair.secretKey, X_bar_c, &ptxXbar);
                result.X_bar_c_decrypted[c] = ptxXbar->GetRealPackedValue();
            }

            if (verbose) {
                std::cout << "    X_bar_c computed. Level: " << X_bar_c->GetLevel() << std::endl;

                std::cout << "    X_bar_c (first 5 rows, first " << f << " cols):" << std::endl;
                for (int row = 0; row < 5 && row < (int)s_c; row++) {
                    std::cout << "      Row " << row << ": ";
                    for (size_t col = 0; col < f; col++) {
                        std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                                  << result.X_bar_c_decrypted[c][row * largeDim + col] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }

            // Compute X_bar_c^T
            if (verbose) {
                std::cout << "    Computing X_bar_c^T (transpose)..." << std::endl;
            }
            auto X_bar_c_T = eval_transpose(X_bar_c, largeDim, largeDim * largeDim);

            if (verbose) {
                std::cout << "    X_bar_c^T computed. Level: " << X_bar_c_T->GetLevel() << std::endl;

                // Print first few rows of X_bar_c^T
                Plaintext ptxXbarT;
                m_cc->Decrypt(m_keyPair.secretKey, X_bar_c_T, &ptxXbarT);
                std::vector<double> xbarTVals = ptxXbarT->GetRealPackedValue();

                std::cout << "    X_bar_c^T (first " << f << " rows, first 5 cols):" << std::endl;
                for (size_t row = 0; row < f; row++) {
                    std::cout << "      Row " << row << ": ";
                    for (int col = 0; col < 5 && col < (int)s_c; col++) {
                        std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                                  << xbarTVals[row * largeDim + col] << " ";
                    }
                    std::cout << "..." << std::endl;
                }
                std::cout << std::endl;
            }

            // S_c = X_bar_c^T * X_bar_c using JKLS18
            if (verbose) {
                std::cout << "    Computing S_c = X_bar_c^T * X_bar_c with JKLS18 (" << largeDim << "x" << largeDim << ")..." << std::endl << std::flush;
            }
            auto S_c = eval_mult_JKLS18(X_bar_c_T, X_bar_c, largeDim);

            if (verbose) {
                // Decrypt S_c only when verbose for debugging
                Plaintext ptxSc256;
                m_cc->Decrypt(m_keyPair.secretKey, S_c, &ptxSc256);
                result.S_c_decrypted[c] = ptxSc256->GetRealPackedValue();
            }

            if (verbose) {
                std::cout << "    S_c computed. Level: " << S_c->GetLevel() << std::endl;

                std::cout << "    S_c (256x256, top-left " << f << "x" << f << " before rebatch):" << std::endl;
                for (size_t row = 0; row < f; row++) {
                    std::cout << "      ";
                    for (size_t col = 0; col < f; col++) {
                        std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                                  << result.S_c_decrypted[c][row * largeDim + col] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                // Print S_c after rebatch (16x16)
                auto S_c_rebatched = rebatchToFeatureSpace(S_c, largeDim, f_tilde);
                debugPrintMatrix("    S_c (class " + std::to_string(c) + " scatter, after rebatch)", S_c_rebatched, f, f, f_tilde);
            }

            m_cc->EvalAddInPlace(Sw, S_c);

            if (verbose) {
                std::cout << "    Accumulated to S_W" << std::endl;
            }
        }

        // Rebatch S_W from 256×256 to 16×16
        if (verbose) {
            std::cout << "  Rebatching S_W from " << largeDim << "x" << largeDim << " to " << f_tilde << "x" << f_tilde << "..." << std::endl;
        }
        auto Sw_rebatched = rebatchToFeatureSpace(Sw, largeDim, f_tilde);

        auto swEnd = high_resolution_clock::now();
        timings.swComputation = swEnd - swStart;

        // Decrypt S_W for debugging
        {
            Plaintext ptxSw;
            m_cc->Decrypt(m_keyPair.secretKey, Sw_rebatched, &ptxSw);
            result.Sw_decrypted = ptxSw->GetRealPackedValue();
            result.Sw_decrypted.resize(f_tilde * f_tilde);
        }

        if (verbose) {
            std::cout << "  S_W computation complete" << std::endl;
            debugPrintMatrix("S_W", Sw_rebatched, f, f, f_tilde);
        }

        // ========== Step 3: Compute S_W^{-1} ==========
        if (verbose) std::cout << "\n[Step 3] Computing S_W^{-1}..." << std::endl;
        auto invStart = high_resolution_clock::now();

        // Upper bound for trace: N * f (for Z-score normalization)
        double traceUpperBound = static_cast<double>(dataset.numSamples) * f;

        // Pass actual feature dimension f for proper identity matrix (zeros at padding positions)
        result.Sw_inv = eval_inverse_impl(Sw_rebatched, f_tilde, inversionIterations, f, traceUpperBound);

        auto invEnd = high_resolution_clock::now();
        timings.inversionTime = invEnd - invStart;

        if (verbose) {
            std::cout << "  Matrix inversion complete (" << inversionIterations << " iterations)" << std::endl;
            debugPrintMatrix("S_W^{-1}", result.Sw_inv, f, f, f_tilde);
        }

        // Decrypt S_W^{-1}
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, result.Sw_inv, &ptx);
        result.Sw_inv_decrypted = ptx->GetRealPackedValue();
        result.Sw_inv_decrypted.resize(f_tilde * f_tilde);

        // ========== Step 4: Compute w = S_W^{-1} * (μ_1 - μ_0) ==========
        if (verbose) std::cout << "[Step 4] Computing w = S_W^{-1} * (mu_1 - mu_0)..." << std::endl;

        // μ_1 - μ_0 (both in f_tilde x f_tilde replicated form)
        auto meanDiff = m_cc->EvalSub(classMeanVec[1], classMeanVec[0]);

        if (verbose) {
            debugPrintVector("(mu_1 - mu_0)", meanDiff, f);
        }

        // w = S_W^{-1} * (μ_1 - μ_0) using matrix-vector multiplication
        // meanDiff is replicated row vector, S_W^{-1} is matrix
        // Result: w as replicated row vector
        int s = std::min((int)f_tilde, (int)(m_cc->GetRingDimension() / 2 / f_tilde / f_tilde));
        s = std::max(1, s);
        int B = f_tilde / s;

        result.Sw_inv_Sb = eval_mult_NewCol(result.Sw_inv, meanDiff, s, B, 4, 4, 4, f_tilde);
        result.Sw_inv_Sb->SetSlots(f_tilde * f_tilde);

        if (verbose) {
            debugPrintVector("w = S_W^{-1} * (mu_1 - mu_0)", result.Sw_inv_Sb, f);
        }

        auto totalEnd = high_resolution_clock::now();
        timings.totalTime = totalEnd - totalStart;

        if (verbose) {
            std::cout << "\n========== Training Complete ==========" << std::endl;
            std::cout << "Mean computation: " << timings.meanComputation.count() << " s" << std::endl;
            std::cout << "S_W computation: " << timings.swComputation.count() << " s" << std::endl;
            std::cout << "Matrix inversion: " << timings.inversionTime.count() << " s" << std::endl;
            std::cout << "Total time: " << timings.totalTime.count() << " s" << std::endl;
        }

        return result;
    }
};
