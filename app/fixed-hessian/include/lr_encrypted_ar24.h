// lr_encrypted_ar24.h
// AR24 matrix multiplication and Schulz-iteration inversion for 16x16 matrices
#pragma once

#include "lr_encrypted_base.h"

class LR_AR24 : public LREncryptedBase {
private:
    // AR24-specific mask generation
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

    // AR24 matrix multiplication: d x d matrices with s copies
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

    // Clean operation: mask to first d*d block
    Ciphertext<DCRTPoly> clean(const Ciphertext<DCRTPoly>& M, int d, int s) {
        std::vector<double> msk(d * d * s, 0.0);
        for (int i = 0; i < d * d; i++) {
            msk[i] = 1.0;
        }
        auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d * s);
        return m_cc->EvalMult(M, pmsk);
    }

public:
    LR_AR24(std::shared_ptr<Encryption> enc,
            CryptoContext<DCRTPoly> cc,
            KeyPair<DCRTPoly> keyPair,
            std::vector<int> rotIndices,
            int multDepth,
            bool useBootstrapping = true)
        : LREncryptedBase(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping)
    {}

    // Matrix inversion using AR24 multiplication + Schulz iteration
    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d,
                                       int iterations, int actualDim,
                                       double traceUpperBound) override {
        int maxBatch = m_cc->GetRingDimension() / 2;
        int s = std::min(d, maxBatch / d / d);
        s = std::max(1, s);
        int num_slots = d * d * s;

        // Identity: 1s only at actual feature positions (0..actualDim-1)
        std::vector<double> vI(d * d, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI[i * d + i] = 1.0;
        }
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);
        auto I_enc = m_cc->Encrypt(m_keyPair.publicKey, pI);

        // Identity for AR24 multiplication (d*d*s slots)
        std::vector<double> vI_mult(num_slots, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI_mult[i * d + i] = 1.0;
        }
        Plaintext pI_mult = m_cc->MakeCKKSPackedPlaintext(vI_mult, 1, 0, nullptr, num_slots);

        // Compute trace(M) in encrypted form
        auto traceEnc = eval_trace(M, d, d * d);

        if (traceUpperBound <= 0) {
            traceUpperBound = 64.0 * actualDim;
        }

        auto alphaEnc = eval_scalar_inverse(traceEnc, traceUpperBound, 2, d * d);

        if (m_verbose) {
            Plaintext ptxTrace, ptxAlpha;
            m_cc->Decrypt(m_keyPair.secretKey, traceEnc, &ptxTrace);
            m_cc->Decrypt(m_keyPair.secretKey, alphaEnc, &ptxAlpha);
            std::cout << "  [Inversion] trace(M) = " << ptxTrace->GetRealPackedValue()[0]
                      << ", alpha = " << ptxAlpha->GetRealPackedValue()[0] << std::endl;
        }

        // Y_0 = alpha * I, A_bar = I - alpha * M
        auto Y = m_cc->EvalMult(I_enc, alphaEnc);
        auto A_bar = m_cc->EvalSub(I_enc, m_cc->EvalMult(M, alphaEnc));

        if (m_verbose) {
            std::cout << "  [Inversion Init] Y level: " << Y->GetLevel()
                      << ", A_bar level: " << A_bar->GetLevel() << std::endl;
        }

        Y->SetSlots(num_slots);
        A_bar->SetSlots(num_slots);
        Y = clean(Y, d, s);
        A_bar = clean(A_bar, d, s);

        for (int i = 0; i < iterations - 1; i++) {
            // AR24 matrix multiplication
            Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI_mult, A_bar), d, s);
            A_bar = eval_mult_AR24(A_bar, A_bar, d, s);

            // Check level and bootstrap if needed
            if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 3) {
                if (m_verbose) {
                    std::cout << "  [Iter " << i << "] Bootstrapping. Y level: "
                              << Y->GetLevel() << std::endl;
                }
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);

                A_bar->SetSlots(d * d * s);
                A_bar = clean(A_bar, d, s);
                Y->SetSlots(d * d * s);
                Y = clean(Y, d, s);
            } else {
                A_bar->SetSlots(num_slots);
                A_bar = clean(A_bar, d, s);
                Y->SetSlots(num_slots);
                Y = clean(Y, d, s);
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
        }

        Y = eval_mult_AR24(Y, m_cc->EvalAdd(pI_mult, A_bar), d, s);
        Y->SetSlots(d * d);

        if (m_verbose) {
            std::cout << "  [Inversion Done] Final Y level: " << Y->GetLevel() << std::endl;
        }

        return Y;
    }
};
