// lda_ar24.h
// LDA implementation using AR24 algorithm for matrix inversion
// NOTE: Only 64 samples supported. Other sample sizes not yet implemented.
#pragma once

#include "lda_encrypted_base.h"

class LDA_AR24 : public LDAEncryptedBase {
private:
    // AR24-specific mask generation for matrix multiplication
    std::vector<double> generatePhiMsk(int k, int d, int s) {
        std::vector<double> msk(d * d * s, 0);
        for (int i = k; i < d * d * s; i += d) {
            msk[i] = 1;
        }
        return msk;
    }

    std::vector<double> generatePsiMsk(int k, int d, int s) {
        std::vector<double> msk(d * d * s, 0);
        for (int i = 0; i < s; i++) {
            for (int j = i * d * d + k * d; j < i * d * d + k * d + d; j++) {
                msk[j] = 1;
            }
        }
        return msk;
    }

    // AR24 matrix multiplication: d×d matrices with s copies
    Ciphertext<DCRTPoly> eval_mult_AR24(const Ciphertext<DCRTPoly>& matA,
                                        const Ciphertext<DCRTPoly>& matB,
                                        int d, int s) {
        int B = d / s;
        int num_slots = d * d * s;

        auto matrixC = getZeroCiphertext(num_slots)->Clone();
        auto matrixA_copy = matA->Clone();
        auto matrixB_copy = matB->Clone();
        matrixA_copy->SetSlots(num_slots);
        matrixB_copy->SetSlots(num_slots);

        std::vector<Ciphertext<DCRTPoly>> Tilde_A(B);
        std::vector<Ciphertext<DCRTPoly>> Tilde_B(B);

        // Preprocessing for A
        for (int i = 0; i < (int)log2(s); i++) {
            auto tmp = rot.rotate(matrixA_copy, (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixA_copy, tmp);
        }

        // Preprocessing for B
        for (int i = 0; i < (int)log2(s); i++) {
            auto tmp = rot.rotate(matrixB_copy, d * (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixB_copy, tmp);
        }

        // Build Tilde_A
        for (int i = 0; i < B; i++) {
            auto phi_si = m_cc->MakeCKKSPackedPlaintext(generatePhiMsk(s * i, d, s), 1, 0, nullptr, num_slots);
            auto tmp = m_cc->EvalMult(matrixA_copy, phi_si);
            tmp = rot.rotate(tmp, s * i);
            for (int j = 0; j < (int)log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j)));
            }
            Tilde_A[i] = tmp;
        }

        // Build Tilde_B
        for (int i = 0; i < B; i++) {
            auto psi_si = m_cc->MakeCKKSPackedPlaintext(generatePsiMsk(s * i, d, s), 1, 0, nullptr, num_slots);
            auto tmp = m_cc->EvalMult(matrixB_copy, psi_si);
            tmp = rot.rotate(tmp, s * i * d);
            for (int j = 0; j < (int)log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j) * d));
            }
            Tilde_B[i] = tmp;
        }

        // Compute C = sum of Tilde_A[i] * Tilde_B[i]
        for (int i = 0; i < B; i++) {
            m_cc->EvalAddInPlace(matrixC,
                m_cc->EvalMultAndRelinearize(Tilde_A[i], Tilde_B[i]));
        }

        // Final accumulation
        for (int i = 0; i < (int)log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC, rot.rotate(matrixC, (d * d) * (1 << i)));
        }

        matrixC->SetSlots(d * d);
        return matrixC;
    }

    // Clean operation: mask to first d×d block
    Ciphertext<DCRTPoly> clean(const Ciphertext<DCRTPoly>& M, int d, int s) {
        std::vector<double> msk(d * d * s, 0.0);
        for (int i = 0; i < d * d; i++) {
            msk[i] = 1.0;
        }
        auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d * s);
        return m_cc->EvalMult(M, pmsk);
    }

public:
    LDA_AR24(std::shared_ptr<Encryption> enc,
             CryptoContext<DCRTPoly> cc,
             KeyPair<DCRTPoly> keyPair,
             std::vector<int> rotIndices,
             int multDepth,
             bool useBootstrapping = true)
        : LDAEncryptedBase(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping)
    {}

    bool m_verbose = false;

    // Scalar inverse using power series: computes 1/t given encrypted t and upper_bound u >= t
    // batchSize: must match trace's slot count
    Ciphertext<DCRTPoly> eval_scalar_inverse(const Ciphertext<DCRTPoly>& t, double upperBound, int iterations, int batchSize) {
        double x0 = 1.0 / upperBound;
        auto x = m_cc->Encrypt(m_keyPair.publicKey,
            m_cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, x0), 1, 0, nullptr, batchSize));
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        if (m_verbose) {
            std::cout << "  [Scalar Inv] upper_bound = " << upperBound << ", x0 = " << x0 << std::endl;
        }

        for (int i = 0; i < iterations; i++) {
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }
        return x;
    }

    // Matrix inversion using AR24 multiplication (same structure as NewCol)
    Ciphertext<DCRTPoly> eval_inverse_impl(const Ciphertext<DCRTPoly>& M, int d, int iterations, int actualDim, double traceUpperBound = 0) {
        int maxBatch = m_cc->GetRingDimension() / 2;
        int s = std::min(d, maxBatch / d / d);
        s = std::max(1, s);
        int num_slots = d * d * s;

        // DEBUG: Check input slots and values
        if (m_verbose) {
            std::cout << "  [DEBUG] Input M->GetSlots() = " << M->GetSlots() << std::endl;
            std::cout << "  [DEBUG] Expected num_slots = " << num_slots << " (d=" << d << ", s=" << s << ")" << std::endl;

            Plaintext ptM;
            m_cc->Decrypt(m_keyPair.secretKey, M, &ptM);
            auto mVec = ptM->GetRealPackedValue();

            // Count non-zero values
            int nonZeroCount = 0;
            double maxVal = 0, minVal = 0;
            for (size_t i = 0; i < std::min(mVec.size(), (size_t)num_slots); i++) {
                if (std::abs(mVec[i]) > 1e-6) nonZeroCount++;
                maxVal = std::max(maxVal, mVec[i]);
                minVal = std::min(minVal, mVec[i]);
            }
            std::cout << "  [DEBUG] M: nonzero=" << nonZeroCount << "/" << num_slots
                      << ", range=[" << minVal << ", " << maxVal << "]" << std::endl;

            // Check values in different slot ranges
            std::cout << "  [DEBUG] M slots 0-255 (first d*d): ";
            for (int i = 0; i < 5; i++) std::cout << mVec[i*d+i] << " ";
            std::cout << "..." << std::endl;

            if (num_slots > d*d) {
                std::cout << "  [DEBUG] M slots 256-511 (second d*d): ";
                for (int i = 0; i < 5; i++) std::cout << mVec[d*d + i*d+i] << " ";
                std::cout << "..." << std::endl;
            }
        }

        // Identity for initial computation (d*d slots, matches trace/alpha)
        // 1s only at actual feature positions (0..actualDim-1), 0s at padding (actualDim..d-1)
        std::vector<double> vI(d * d, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI[i * d + i] = 1.0;
        }
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);
        auto I_enc = m_cc->Encrypt(m_keyPair.publicKey, pI);

        // Identity for AR24 multiplication (d*d*s slots, for I + A_bar)
        // Same pattern: 1s at 0..actualDim-1 diagonal, 0s elsewhere
        std::vector<double> vI_mult(num_slots, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI_mult[i * d + i] = 1.0;
        }
        Plaintext pI_mult = m_cc->MakeCKKSPackedPlaintext(vI_mult, 1, 0, nullptr, num_slots);

        if (m_verbose && actualDim < d) {
            std::cout << "  [Inversion] Identity: 1s at 0.." << actualDim-1 << ", 0s at " << actualDim << ".." << d-1 << std::endl;
        }

        // Compute trace(M) in encrypted form (d*d slots)
        auto traceEnc = eval_trace(M, d, d * d);

        if (traceUpperBound <= 0) {
            traceUpperBound = actualDim * actualDim;
        }

        // Compute encrypted alpha = 1/trace using power series (d*d slots, same as trace)
        auto alphaEnc = eval_scalar_inverse(traceEnc, traceUpperBound, 3, d * d);

        if (m_verbose) {
            Plaintext ptxTrace, ptxAlpha;
            m_cc->Decrypt(m_keyPair.secretKey, traceEnc, &ptxTrace);
            m_cc->Decrypt(m_keyPair.secretKey, alphaEnc, &ptxAlpha);
            auto traceVec = ptxTrace->GetRealPackedValue();
            auto alphaVec = ptxAlpha->GetRealPackedValue();
            std::cout << "  [Inversion] trace(M) = " << traceVec[0]
                      << ", alpha = " << alphaVec[0] << std::endl;
            // Check if trace/alpha are consistent across all d*d slots
            bool consistent = true;
            for (int i = 1; i < d * d && i < (int)traceVec.size(); i++) {
                if (std::abs(traceVec[i] - traceVec[0]) > 0.01 || std::abs(alphaVec[i] - alphaVec[0]) > 0.01) {
                    consistent = false;
                    break;
                }
            }
            if (!consistent) {
                std::cout << "  [WARNING] trace/alpha NOT consistent across slots!" << std::endl;
            }
        }

        // Y_0 = alpha * I, A_bar = I - alpha * M
        auto Y = m_cc->EvalMult(I_enc, alphaEnc);
        auto A_bar = m_cc->EvalSub(I_enc, m_cc->EvalMult(M, alphaEnc));

        if (m_verbose) {
            std::cout << "  [Inversion Init] Y level: " << Y->GetLevel()
                      << ", A_bar level: " << A_bar->GetLevel() << std::endl;
            // Print Y and A_bar diagonal values before SetSlots
            Plaintext ptY, ptA;
            m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY);
            m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA);
            auto yVec = ptY->GetRealPackedValue();
            auto aVec = ptA->GetRealPackedValue();
            std::cout << "  [Init] Y diagonal (0-12): ";
            for (int i = 0; i < actualDim && i < 5; i++) std::cout << yVec[i*d+i] << " ";
            std::cout << "..." << std::endl;
            std::cout << "  [Init] A_bar diagonal (0-12): ";
            for (int i = 0; i < actualDim && i < 5; i++) std::cout << aVec[i*d+i] << " ";
            std::cout << "..." << std::endl;
        }

        Y->SetSlots(num_slots);
        A_bar->SetSlots(num_slots);
        Y = clean(Y, d, s);
        A_bar = clean(A_bar, d, s);

        // DEBUG: Check after initial SetSlots + clean
        if (m_verbose) {
            std::cout << "  [DEBUG] After SetSlots(" << num_slots << ") + clean:" << std::endl;
            std::cout << "    Y->GetSlots()=" << Y->GetSlots() << ", A_bar->GetSlots()=" << A_bar->GetSlots() << std::endl;

            Plaintext ptY, ptA;
            m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY);
            m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA);
            auto yVec = ptY->GetRealPackedValue();
            auto aVec = ptA->GetRealPackedValue();

            // Check all slots
            int yNonZero = 0, aNonZero = 0;
            for (size_t j = 0; j < std::min(yVec.size(), (size_t)num_slots); j++) {
                if (std::abs(yVec[j]) > 1e-9) yNonZero++;
                if (std::abs(aVec[j]) > 1e-9) aNonZero++;
            }
            std::cout << "    Y nonzero slots: " << yNonZero << "/" << num_slots << std::endl;
            std::cout << "    A_bar nonzero slots: " << aNonZero << "/" << num_slots << std::endl;

            // Check slots 256-511 (should be 0 after clean)
            double yMax256 = 0, aMax256 = 0;
            for (int j = d*d; j < std::min(2*d*d, num_slots); j++) {
                yMax256 = std::max(yMax256, std::abs(yVec[j]));
                aMax256 = std::max(aMax256, std::abs(aVec[j]));
            }
            std::cout << "    Slots 256-511 max: Y=" << yMax256 << ", A_bar=" << aMax256 << std::endl;
        }

        for (int i = 0; i < iterations - 1; i++) {
            // AR24 matrix multiplication (FIRST, like src)
            Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI_mult, A_bar), d, s);
            A_bar = eval_mult_AR24(A_bar, A_bar, d, s);

            // Check level and bootstrap/clean AFTER multiplication (like src)
            // DEBUG: Force bootstrap at iteration 3 to test bootstrap behavior
            bool forceBootstrap = (i == 3);
            if (m_useBootstrapping && (forceBootstrap || (int)Y->GetLevel() >= m_multDepth - 3)) {
                // BEFORE bootstrap
                if (m_verbose) {
                    Plaintext ptY_pre, ptA_pre;
                    m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY_pre);
                    m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA_pre);
                    auto yPre = ptY_pre->GetRealPackedValue();
                    auto aPre = ptA_pre->GetRealPackedValue();
                    std::cout << "  [Iter " << i << "] BEFORE bootstrap (level=" << Y->GetLevel() << "):" << std::endl;
                    std::cout << "    Y[0,0]=" << yPre[0] << " Y[1,1]=" << yPre[d+1] << " Y[2,2]=" << yPre[2*d+2] << std::endl;
                    std::cout << "    A[0,0]=" << aPre[0] << " A[1,1]=" << aPre[d+1] << " A[2,2]=" << aPre[2*d+2] << std::endl;
                }

                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);

                // AFTER bootstrap, BEFORE clean
                if (m_verbose) {
                    Plaintext ptY_post, ptA_post;
                    m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY_post);
                    m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA_post);
                    auto yPost = ptY_post->GetRealPackedValue();
                    auto aPost = ptA_post->GetRealPackedValue();
                    std::cout << "  [Iter " << i << "] AFTER bootstrap (level=" << Y->GetLevel() << "):" << std::endl;
                    std::cout << "    Y[0,0]=" << yPost[0] << " Y[1,1]=" << yPost[d+1] << " Y[2,2]=" << yPost[2*d+2] << std::endl;
                    std::cout << "    A[0,0]=" << aPost[0] << " A[1,1]=" << aPost[d+1] << " A[2,2]=" << aPost[2*d+2] << std::endl;
                }

                A_bar->SetSlots(d * d * s);
                A_bar = clean(A_bar, d, s);
                Y->SetSlots(d * d * s);
                Y = clean(Y, d, s);

                // AFTER clean
                if (m_verbose) {
                    Plaintext ptY_clean, ptA_clean;
                    m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY_clean);
                    m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA_clean);
                    auto yClean = ptY_clean->GetRealPackedValue();
                    auto aClean = ptA_clean->GetRealPackedValue();
                    std::cout << "  [Iter " << i << "] AFTER clean (level=" << Y->GetLevel() << "):" << std::endl;
                    std::cout << "    Y[0,0]=" << yClean[0] << " Y[1,1]=" << yClean[d+1] << " Y[2,2]=" << yClean[2*d+2] << std::endl;
                    std::cout << "    A[0,0]=" << aClean[0] << " A[1,1]=" << aClean[d+1] << " A[2,2]=" << aClean[2*d+2] << std::endl;
                }
            } else {
                // Must restore slot count before clean (eval_mult_AR24 sets to d*d)
                A_bar->SetSlots(num_slots);
                A_bar = clean(A_bar, d, s);
                Y->SetSlots(num_slots);
                Y = clean(Y, d, s);
            }

            if (m_verbose && i < 3) {
                // Debug: print Y and A_bar diagonal after iteration
                Plaintext ptY, ptA;
                m_cc->Decrypt(m_keyPair.secretKey, Y, &ptY);
                m_cc->Decrypt(m_keyPair.secretKey, A_bar, &ptA);
                auto yVec = ptY->GetRealPackedValue();
                auto aVec = ptA->GetRealPackedValue();
                std::cout << "  [Iter " << i << "] Y diag: ";
                for (int j = 0; j < 5; j++) std::cout << std::setprecision(6) << yVec[j*d+j] << " ";
                std::cout << std::endl;
                std::cout << "  [Iter " << i << "] A_bar diag: ";
                for (int j = 0; j < 5; j++) std::cout << std::setprecision(6) << aVec[j*d+j] << " ";
                std::cout << std::endl;

                // Check all slots after first iteration
                if (i == 0) {
                    int yNonZero = 0, aNonZero = 0;
                    for (size_t j = 0; j < std::min(yVec.size(), (size_t)num_slots); j++) {
                        if (std::abs(yVec[j]) > 1e-9) yNonZero++;
                        if (std::abs(aVec[j]) > 1e-9) aNonZero++;
                    }
                    std::cout << "  [Iter 0] Full check: Y nonzero=" << yNonZero << ", A_bar nonzero=" << aNonZero << std::endl;
                    std::cout << "  [Iter 0] Y->GetSlots()=" << Y->GetSlots() << ", A_bar->GetSlots()=" << A_bar->GetSlots() << std::endl;

                    // Check slots 256-511
                    double yMax256 = 0, aMax256 = 0;
                    for (int j = d*d; j < std::min(2*d*d, num_slots); j++) {
                        yMax256 = std::max(yMax256, std::abs(yVec[j]));
                        aMax256 = std::max(aMax256, std::abs(aVec[j]));
                    }
                    std::cout << "  [Iter 0] Slots 256-511 max: Y=" << yMax256 << ", A_bar=" << aMax256 << std::endl;
                }
            }

            if (m_verbose && (i % 5 == 0 || i == iterations - 2)) {
                std::cout << "  [Iter " << i << "] Y level: " << Y->GetLevel()
                          << ", A_bar level: " << A_bar->GetLevel() << std::endl;
            }
        }

        // Final multiplication
        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 3) {
            if (m_verbose) {
                std::cout << "  [Before Final] Bootstrapping. Y level: " << Y->GetLevel() << std::endl;
            }
            A_bar->SetSlots(d * d);
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y->SetSlots(d * d);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
            A_bar->SetSlots(d * d * s);
            Y->SetSlots(d * d * s);
            if (m_verbose) {
                std::cout << "           After bootstrap. Y level: " << Y->GetLevel() << std::endl;
            }
        }

        Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI_mult, A_bar), d, s);
        Y->SetSlots(d * d);

        if (m_verbose) {
            std::cout << "  [Inversion Done] Final Y level: " << Y->GetLevel() << std::endl;
        }

        return Y;
    }

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d, int iterations) override {
        return eval_inverse_impl(M, d, iterations, d);
    }

    // Binary LDA training: w = S_W^{-1} * (μ_1 - μ_0)
    // NOTE: Only 64 samples supported
    LDAEncryptedResult trainWithTimings(
        const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
        const LDADataset& dataset,
        int inversionIterations,
        LDATimingResult& timings,
        bool verbose = false,
        bool sbOnly = false) override
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
            std::cout << "\n========== LDA Training (AR24 Encrypted) ==========" << std::endl;
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

            classMeanForSw[c] = eval_computeMeanForSw(classDataEncrypted[c], s_c, f, largeDim);
            classMeanVec[c] = eval_computeMeanForSb(classDataEncrypted[c], s_c, f_tilde, largeDim);
            result.classMeansEncrypted[c] = classMeanVec[c]->Clone();

            Plaintext ptx;
            m_cc->Decrypt(m_keyPair.secretKey, classMeanVec[c], &ptx);
            std::vector<double> meanVec = ptx->GetRealPackedValue();
            result.classMeans[c] = std::vector<double>(meanVec.begin(), meanVec.begin() + f);

            if (verbose) {
                debugPrintLevel("Class " + std::to_string(c) + " mean (for S_W)", classMeanForSw[c]);
                debugPrintVector("Class " + std::to_string(c) + " Mean", classMeanVec[c], f);
            }
        }

        auto globalMeanEnc = eval_computeGlobalMean(classMeanVec, dataset.samplesPerClass, f_tilde);
        result.globalMeanEncrypted = globalMeanEnc->Clone();

        {
            Plaintext ptxGlobal;
            m_cc->Decrypt(m_keyPair.secretKey, globalMeanEnc, &ptxGlobal);
            std::vector<double> globalMeanVec = ptxGlobal->GetRealPackedValue();
            result.globalMean = std::vector<double>(globalMeanVec.begin(), globalMeanVec.begin() + f);
        }

        if (verbose) {
            std::cout << "=== Global Mean (len=" << f << ") ===" << std::endl;
            for (size_t i = 0; i < f; i++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed << result.globalMean[i] << " ";
            }
            std::cout << std::endl << std::endl;
        }

        auto meanEnd = high_resolution_clock::now();
        timings.meanComputation = meanEnd - meanStart;

        // ========== Step 2: Compute S_W (Within-class scatter) ==========
        if (verbose) std::cout << "[Step 2] Computing S_W (within-class scatter)..." << std::endl;
        auto swStart = high_resolution_clock::now();

        auto Sw = getZeroCiphertext(largeDim * largeDim);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];

            if (verbose) {
                std::cout << "  Class " << c << " (n=" << s_c << "):" << std::endl;
            }

            auto X_bar_c = m_cc->EvalSub(classDataEncrypted[c], classMeanForSw[c]);
            auto X_bar_c_T = eval_transpose(X_bar_c, largeDim, largeDim * largeDim);

            if (verbose) {
                std::cout << "    Computing S_c = X_bar_c^T * X_bar_c with JKLS18..." << std::endl;
            }
            auto S_c = eval_mult_JKLS18(X_bar_c_T, X_bar_c, largeDim);

            m_cc->EvalAddInPlace(Sw, S_c);
        }

        auto Sw_rebatched = rebatchToFeatureSpace(Sw, largeDim, f_tilde);

        auto swEnd = high_resolution_clock::now();
        timings.swComputation = swEnd - swStart;

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

        double traceUpperBound = static_cast<double>(dataset.numSamples) * f;
        result.Sw_inv = eval_inverse_impl(Sw_rebatched, f_tilde, inversionIterations, f, traceUpperBound);

        auto invEnd = high_resolution_clock::now();
        timings.inversionTime = invEnd - invStart;

        if (verbose) {
            std::cout << "  Matrix inversion complete (" << inversionIterations << " iterations)" << std::endl;
            debugPrintMatrix("S_W^{-1}", result.Sw_inv, f, f, f_tilde);
        }

        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, result.Sw_inv, &ptx);
        result.Sw_inv_decrypted = ptx->GetRealPackedValue();
        result.Sw_inv_decrypted.resize(f_tilde * f_tilde);

        // ========== Step 4: Compute w = S_W^{-1} * (μ_1 - μ_0) ==========
        if (verbose) std::cout << "[Step 4] Computing w = S_W^{-1} * (mu_1 - mu_0)..." << std::endl;

        auto meanDiff = m_cc->EvalSub(classMeanVec[1], classMeanVec[0]);

        if (verbose) {
            debugPrintVector("(mu_1 - mu_0)", meanDiff, f);
        }

        int s = std::min((int)f_tilde, (int)(m_cc->GetRingDimension() / 2 / f_tilde / f_tilde));
        s = std::max(1, s);

        result.Sw_inv_Sb = eval_mult_AR24(result.Sw_inv, meanDiff, f_tilde, s);
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
