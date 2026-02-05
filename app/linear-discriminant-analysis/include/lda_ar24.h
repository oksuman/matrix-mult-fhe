// lda_ar24.h
// LDA implementation using AR24 algorithm for matrix inversion
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

    // AR24 matrix inversion using Schulz iteration
    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d, int iterations) override {
        int s = std::min(d, (int)(m_cc->GetRingDimension() / 2 / d / d));
        s = std::max(1, s);

        std::vector<double> vI = initializeIdentityMatrix(d);
        std::vector<double> vI2 = initializeIdentityMatrix2(d, d * d * s);
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);
        Plaintext pI2 = m_cc->MakeCKKSPackedPlaintext(vI2, 1, 0, nullptr, d * d * s);

        // Compute trace for initial scaling
        auto trace = eval_trace(M, d, d * d);

        // For HD dataset with 16×16 matrix, trace is sum of diagonal elements
        // Estimate range for EvalDivide based on matrix properties
        int traceMin = (d * d) / 3 - d;
        int traceMax = (d * d) / 3 + d;
        auto trace_reciprocal = m_cc->EvalDivide(trace, traceMin, traceMax, 50);

        // Y_0 = (1/trace) * I
        auto Y = m_cc->EvalMult(pI, trace_reciprocal);
        // A_bar = I - M * Y_0 = I - M/trace
        auto A_bar = m_cc->EvalSub(pI, m_cc->EvalMultAndRelinearize(M, trace_reciprocal));

        // Extend to s copies for AR24 multiplication
        Y->SetSlots(d * d * s);
        A_bar->SetSlots(d * d * s);
        Y = clean(Y, d, s);
        A_bar = clean(A_bar, d, s);

        // Schulz iteration: Y = Y * (I + A_bar), A_bar = A_bar^2
        for (int i = 0; i < iterations - 1; i++) {
            Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI2, A_bar), d, s);
            A_bar = eval_mult_AR24(A_bar, A_bar, d, s);

            if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 3) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);

                A_bar->SetSlots(d * d * s);
                A_bar = clean(A_bar, d, s);
                Y->SetSlots(d * d * s);
                Y = clean(Y, d, s);
            } else {
                A_bar = clean(A_bar, d, s);
                Y = clean(Y, d, s);
            }
        }

        // Final iteration
        Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI2, A_bar), d, s);
        Y->SetSlots(d * d);

        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 3) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }

        return Y;
    }

    // Main LDA training
    LDAEncryptedResult trainWithTimings(
        const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
        const LDADataset& dataset,
        int inversionIterations,
        LDATimingResult& timings,
        bool verbose = false) override
    {
        using namespace std::chrono;
        auto totalStart = high_resolution_clock::now();

        LDAEncryptedResult result;
        size_t f = dataset.numFeatures;            // 13 for HD
        size_t f_tilde = dataset.paddedFeatures;   // 16
        size_t s_tilde = dataset.paddedSamples;    // 256
        size_t numClasses = dataset.numClasses;    // 2
        int largeDim = HD_MATRIX_DIM;              // 256 (max(s̃, f̃))

        result.classCounts = dataset.samplesPerClass;
        result.classMeans.resize(numClasses);

        if (verbose) {
            std::cout << "\n========== LDA Training (AR24 Encrypted) ==========" << std::endl;
            std::cout << "Features: " << f << " (padded: " << f_tilde << ")" << std::endl;
            std::cout << "Samples: " << dataset.numSamples << " (padded: " << s_tilde << ")" << std::endl;
            std::cout << "Matrix dimension for JKLS18: " << largeDim << "x" << largeDim << std::endl;
            std::cout << "Bootstrapping: " << (m_useBootstrapping ? "enabled" : "disabled") << std::endl;
        }

        // ========== Step 1: Compute Class Means ==========
        if (verbose) std::cout << "\n[Step 1] Computing class means..." << std::endl;
        auto meanStart = high_resolution_clock::now();

        result.classMeansEncrypted.resize(numClasses);
        std::vector<Ciphertext<DCRTPoly>> classMeanReplicated(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_c = dataset.samplesPerClass[c];
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            // Compute mean via folding sum
            classMeanReplicated[c] = eval_computeMean(classDataEncrypted[c], s_c, s_tilde_c, f_tilde);
            result.classMeansEncrypted[c] = classMeanReplicated[c]->Clone();

            // Decrypt for plaintext inference
            Plaintext ptx;
            m_cc->Decrypt(m_keyPair.secretKey, classMeanReplicated[c], &ptx);
            std::vector<double> meanVec = ptx->GetRealPackedValue();
            result.classMeans[c] = std::vector<double>(meanVec.begin(), meanVec.begin() + f);

            if (verbose) {
                std::cout << "  Class " << c << " mean (first 5): ";
                for (int i = 0; i < std::min(5, (int)f); i++) {
                    std::cout << std::setprecision(4) << result.classMeans[c][i] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Compute global mean from all samples
        // We need to encrypt all samples first, or compute from class means
        // For simplicity, compute as weighted average of class means
        result.globalMean.resize(f, 0.0);
        size_t totalSamples = 0;
        for (size_t c = 0; c < numClasses; c++) {
            totalSamples += dataset.samplesPerClass[c];
            for (size_t i = 0; i < f; i++) {
                result.globalMean[i] += result.classMeans[c][i] * dataset.samplesPerClass[c];
            }
        }
        for (size_t i = 0; i < f; i++) {
            result.globalMean[i] /= totalSamples;
        }

        auto meanEnd = high_resolution_clock::now();
        timings.meanComputation = meanEnd - meanStart;

        // ========== Step 2: Compute S_W (Within-class scatter) ==========
        // S_W = sum_c (X_c - μ_c)^T * (X_c - μ_c)
        if (verbose) std::cout << "[Step 2] Computing S_W (within-class scatter)..." << std::endl;
        auto swStart = high_resolution_clock::now();

        // Initialize S_W as zeros
        auto Sw = getZeroCiphertext(largeDim * largeDim);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            // Replicate mean to match data shape
            auto meanRep = eval_replicateMean(classMeanReplicated[c], f_tilde, s_tilde_c * f_tilde);

            // X_bar_c = X_c - μ_c
            auto X_bar_c = m_cc->EvalSub(classDataEncrypted[c], meanRep);

            // For JKLS18: need to reshape X_bar_c into largeDim × largeDim matrix
            // X_bar_c is s_tilde_c × f_tilde, embed into largeDim × largeDim
            X_bar_c->SetSlots(largeDim * largeDim);

            // Compute X_bar_c^T
            auto X_bar_c_T = eval_transpose(X_bar_c, largeDim, largeDim * largeDim);

            // S_c = X_bar_c^T * X_bar_c using JKLS18
            auto S_c = eval_mult_JKLS18(X_bar_c_T, X_bar_c, largeDim);

            // Accumulate
            m_cc->EvalAddInPlace(Sw, S_c);
        }

        // Rebatch S_W from largeDim×largeDim to f_tilde×f_tilde
        auto Sw_rebatched = rebatchToFeatureSpace(Sw, largeDim, f_tilde);

        auto swEnd = high_resolution_clock::now();
        timings.swComputation = swEnd - swStart;

        if (verbose) {
            std::cout << "  S_W computation complete" << std::endl;
            debugPrint("S_W (first 16 elements)", Sw_rebatched, 16);
        }

        // ========== Step 3: Compute S_B (Between-class scatter) ==========
        // S_B = sum_c s_c * (μ_c - μ)(μ_c - μ)^T
        if (verbose) std::cout << "[Step 3] Computing S_B (between-class scatter)..." << std::endl;
        auto sbStart = high_resolution_clock::now();

        auto Sb = getZeroCiphertext(f_tilde * f_tilde);

        // Encrypt global mean
        std::vector<double> globalMeanPadded(f_tilde, 0.0);
        for (size_t i = 0; i < f; i++) {
            globalMeanPadded[i] = result.globalMean[i];
        }
        auto globalMeanEnc = m_enc->encryptInput(globalMeanPadded);
        result.globalMeanEncrypted = globalMeanEnc->Clone();

        for (size_t c = 0; c < numClasses; c++) {
            // diff = μ_c - μ
            auto diff = m_cc->EvalSub(classMeanReplicated[c], globalMeanEnc);

            // outer = diff * diff^T
            auto outer = eval_outerProduct(diff, f, f_tilde);

            // Scale by class size
            auto scaled = m_cc->EvalMult(outer, (double)dataset.samplesPerClass[c]);

            // Accumulate
            m_cc->EvalAddInPlace(Sb, scaled);
        }

        auto sbEnd = high_resolution_clock::now();
        timings.sbComputation = sbEnd - sbStart;

        if (verbose) {
            std::cout << "  S_B computation complete" << std::endl;
            debugPrint("S_B (first 16 elements)", Sb, 16);
        }

        // ========== Step 4: Compute S_W^{-1} ==========
        if (verbose) std::cout << "[Step 4] Computing S_W^{-1} (AR24 Schulz iteration)..." << std::endl;
        auto invStart = high_resolution_clock::now();

        result.Sw_inv = eval_inverse(Sw_rebatched, f_tilde, inversionIterations);

        auto invEnd = high_resolution_clock::now();
        timings.inversionTime = invEnd - invStart;

        if (verbose) {
            std::cout << "  Matrix inversion complete (" << inversionIterations << " iterations)" << std::endl;
            debugPrint("S_W^{-1} (first 16 elements)", result.Sw_inv, 16);
        }

        // Decrypt S_W^{-1} for plaintext inference
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, result.Sw_inv, &ptx);
        result.Sw_inv_decrypted = ptx->GetRealPackedValue();
        result.Sw_inv_decrypted.resize(f_tilde * f_tilde);

        // ========== Step 5: Compute S_W^{-1} * S_B (optional) ==========
        if (verbose) std::cout << "[Step 5] Computing S_W^{-1} * S_B..." << std::endl;

        // For f_tilde×f_tilde, use direct JKLS18 or AR24 multiplication
        int invS = std::min((int)f_tilde, (int)(m_cc->GetRingDimension() / 2 / f_tilde / f_tilde));
        invS = std::max(1, invS);

        result.Sw_inv_Sb = eval_mult_AR24(result.Sw_inv, Sb, f_tilde, invS);
        result.Sw_inv_Sb->SetSlots(f_tilde * f_tilde);

        auto totalEnd = high_resolution_clock::now();
        timings.totalTime = totalEnd - totalStart;

        if (verbose) {
            std::cout << "\n========== Training Complete ==========" << std::endl;
            std::cout << "Mean computation: " << timings.meanComputation.count() << " s" << std::endl;
            std::cout << "S_W computation: " << timings.swComputation.count() << " s" << std::endl;
            std::cout << "S_B computation: " << timings.sbComputation.count() << " s" << std::endl;
            std::cout << "Matrix inversion: " << timings.inversionTime.count() << " s" << std::endl;
            std::cout << "Total time: " << timings.totalTime.count() << " s" << std::endl;
        }

        return result;
    }
};
