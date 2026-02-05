// lda_newcol.h
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

    // NewCol matrix inversion using Schulz iteration
    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d, int iterations) override {
        // NewCol parameters for d×d matrix
        int maxBatch = m_cc->GetRingDimension() / 2;
        int s = std::min(d, maxBatch / d / d);
        s = std::max(1, s);

        int B = d / s;
        int ng = 2;   // Giant step count
        int nb = 4;   // Baby step count for A
        int np = 2;   // Baby step count for B

        // Adjust parameters based on matrix size
        if (d >= 16) {
            ng = 4;
            nb = 4;
            np = 4;
        }

        std::vector<double> vI = initializeIdentityMatrix(d);
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);

        // Compute trace for initial scaling
        auto trace = eval_trace(M, d, d * d);

        // Estimate trace range
        int traceMin = (d * d) / 3 - d;
        int traceMax = (d * d) / 3 + d;
        auto trace_reciprocal = m_cc->EvalDivide(trace, traceMin, traceMax, 50);

        // Y_0 = (1/trace) * I
        auto Y = m_cc->EvalMult(pI, trace_reciprocal);
        // A_bar = I - M/trace
        auto A_bar = m_cc->EvalSub(pI, m_cc->EvalMultAndRelinearize(M, trace_reciprocal));

        // Schulz iteration
        for (int i = 0; i < iterations - 1; i++) {
            if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }

            Y = eval_mult_NewCol(Y, m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
            A_bar = eval_mult_NewCol(A_bar, A_bar, s, B, ng, nb, np, d);
        }

        // Final iteration
        Y = eval_mult_NewCol(Y, m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);

        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }

        return Y;
    }

    // Main LDA training (same workflow as AR24, different inversion)
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
        size_t f = dataset.numFeatures;
        size_t f_tilde = dataset.paddedFeatures;
        size_t s_tilde = dataset.paddedSamples;
        size_t numClasses = dataset.numClasses;
        int largeDim = HD_MATRIX_DIM;

        result.classCounts = dataset.samplesPerClass;
        result.classMeans.resize(numClasses);

        if (verbose) {
            std::cout << "\n========== LDA Training (NewCol Encrypted) ==========" << std::endl;
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

            classMeanReplicated[c] = eval_computeMean(classDataEncrypted[c], s_c, s_tilde_c, f_tilde);
            result.classMeansEncrypted[c] = classMeanReplicated[c]->Clone();

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

        // Compute global mean
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

        // ========== Step 2: Compute S_W ==========
        if (verbose) std::cout << "[Step 2] Computing S_W (within-class scatter)..." << std::endl;
        auto swStart = high_resolution_clock::now();

        auto Sw = getZeroCiphertext(largeDim * largeDim);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];

            auto meanRep = eval_replicateMean(classMeanReplicated[c], f_tilde, s_tilde_c * f_tilde);
            auto X_bar_c = m_cc->EvalSub(classDataEncrypted[c], meanRep);
            X_bar_c->SetSlots(largeDim * largeDim);

            auto X_bar_c_T = eval_transpose(X_bar_c, largeDim, largeDim * largeDim);
            auto S_c = eval_mult_JKLS18(X_bar_c_T, X_bar_c, largeDim);

            m_cc->EvalAddInPlace(Sw, S_c);
        }

        auto Sw_rebatched = rebatchToFeatureSpace(Sw, largeDim, f_tilde);

        auto swEnd = high_resolution_clock::now();
        timings.swComputation = swEnd - swStart;

        if (verbose) {
            std::cout << "  S_W computation complete" << std::endl;
            debugPrint("S_W (first 16 elements)", Sw_rebatched, 16);
        }

        // ========== Step 3: Compute S_B ==========
        if (verbose) std::cout << "[Step 3] Computing S_B (between-class scatter)..." << std::endl;
        auto sbStart = high_resolution_clock::now();

        auto Sb = getZeroCiphertext(f_tilde * f_tilde);

        std::vector<double> globalMeanPadded(f_tilde, 0.0);
        for (size_t i = 0; i < f; i++) {
            globalMeanPadded[i] = result.globalMean[i];
        }
        auto globalMeanEnc = m_enc->encryptInput(globalMeanPadded);
        result.globalMeanEncrypted = globalMeanEnc->Clone();

        for (size_t c = 0; c < numClasses; c++) {
            auto diff = m_cc->EvalSub(classMeanReplicated[c], globalMeanEnc);
            auto outer = eval_outerProduct(diff, f, f_tilde);
            auto scaled = m_cc->EvalMult(outer, (double)dataset.samplesPerClass[c]);
            m_cc->EvalAddInPlace(Sb, scaled);
        }

        auto sbEnd = high_resolution_clock::now();
        timings.sbComputation = sbEnd - sbStart;

        if (verbose) {
            std::cout << "  S_B computation complete" << std::endl;
            debugPrint("S_B (first 16 elements)", Sb, 16);
        }

        // ========== Step 4: Compute S_W^{-1} (NewCol) ==========
        if (verbose) std::cout << "[Step 4] Computing S_W^{-1} (NewCol Schulz iteration)..." << std::endl;
        auto invStart = high_resolution_clock::now();

        result.Sw_inv = eval_inverse(Sw_rebatched, f_tilde, inversionIterations);

        auto invEnd = high_resolution_clock::now();
        timings.inversionTime = invEnd - invStart;

        if (verbose) {
            std::cout << "  Matrix inversion complete (" << inversionIterations << " iterations)" << std::endl;
            debugPrint("S_W^{-1} (first 16 elements)", result.Sw_inv, 16);
        }

        // Decrypt S_W^{-1}
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, result.Sw_inv, &ptx);
        result.Sw_inv_decrypted = ptx->GetRealPackedValue();
        result.Sw_inv_decrypted.resize(f_tilde * f_tilde);

        // ========== Step 5: Compute S_W^{-1} * S_B ==========
        if (verbose) std::cout << "[Step 5] Computing S_W^{-1} * S_B..." << std::endl;

        int s = std::min((int)f_tilde, (int)(m_cc->GetRingDimension() / 2 / f_tilde / f_tilde));
        s = std::max(1, s);
        int B = f_tilde / s;

        result.Sw_inv_Sb = eval_mult_NewCol(result.Sw_inv, Sb, s, B, 4, 4, 4, f_tilde);
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
