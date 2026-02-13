// lr_encrypted_newcol.h
// NewCol matrix multiplication and Schulz-iteration inversion for 16x16 matrices
#pragma once

#include "lr_encrypted_base.h"

class LR_NewCol : public LREncryptedBase {
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

        // Baby steps for A
        for (int i = 0; i < nb; i++) {
            babyStepsOfA[i] = rot.rotate(matrixA, i);
        }

        // Baby steps for B
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
    LR_NewCol(std::shared_ptr<Encryption> enc,
              CryptoContext<DCRTPoly> cc,
              KeyPair<DCRTPoly> keyPair,
              std::vector<int> rotIndices,
              int multDepth,
              bool useBootstrapping = true)
        : LREncryptedBase(enc, cc, keyPair, rotIndices, multDepth, useBootstrapping)
    {}

    // Matrix inversion using NewCol multiplication + Schulz iteration
    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d,
                                       int iterations, int actualDim,
                                       double traceUpperBound) override {
        int maxBatch = m_cc->GetRingDimension() / 2;
        int s = std::min(d, maxBatch / d / d);
        s = std::max(1, s);

        int B = d / s;
        int ng = 4;
        int nb = 4;
        int np = 4;

        // Identity matrix: 1s only at actual feature positions
        std::vector<double> vI(d * d, 0.0);
        for (int i = 0; i < actualDim; i++) {
            vI[i * d + i] = 1.0;
        }
        Plaintext pI = m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);
        auto I_enc = m_cc->Encrypt(m_keyPair.publicKey, pI);

        // Compute trace(M) in encrypted form
        auto traceEnc = eval_trace(M, d, d * d);

        if (traceUpperBound <= 0) {
            traceUpperBound = 64.0 * actualDim;
        }

        auto alphaEnc = eval_scalar_inverse(traceEnc, traceUpperBound, 3, d * d);

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

        for (int i = 0; i < iterations - 1; i++) {
            if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
                if (m_verbose) {
                    std::cout << "  [Iter " << i << "] Bootstrapping. Y level: "
                              << Y->GetLevel() << std::endl;
                }
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
                if (m_verbose) {
                    std::cout << "           After bootstrap. Y level: " << Y->GetLevel() << std::endl;
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
        }

        Y = eval_mult_NewCol(Y, m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);

        if (m_useBootstrapping && (int)Y->GetLevel() >= m_multDepth - 2) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }

        if (m_verbose) {
            std::cout << "  [Inversion Done] Final Y level: " << Y->GetLevel() << std::endl;
        }

        return Y;
    }
};
