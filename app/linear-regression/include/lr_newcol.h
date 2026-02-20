#pragma once

#include "lr_base.h"

class LinearRegression_NewCol : public LinearRegressionBase {
private:
    const int m_maxBatch;
    const int m_multDepth;

    std::vector<double> generateMaskVector(int batch_size, int k, int d) {
        std::vector<double> result(batch_size, 0.0);
        for (int i = k * d * d; 
             i < (k + 1) * d * d; ++i) {
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

    Ciphertext<DCRTPoly>
    vecRotsOpt(const std::vector<Ciphertext<DCRTPoly>> &matrixM, int is, int s, int np, int d) {
        auto rotsM = this->getZeroCiphertext(d)->Clone();
        for (int j = 0; j < s / np; j++) {

            auto T = this->getZeroCiphertext(d)->Clone();

            for (int i = 0; i < np; i++) {
                auto msk = generateMaskVector(d*d*s, np * j + i, d);
                msk = vectorRotate(msk, -is * d * s - j * d * np);

                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr,
                                                          d*d*s);
                m_cc->EvalAddInPlace(T, m_cc->EvalMult(matrixM[i], pmsk));
            }
            m_cc->EvalAddInPlace(rotsM, rot.rotate(T, is * d * s + j * d * np));
        }

        return rotsM;
    }

    Ciphertext<DCRTPoly> vecRots(const Ciphertext<DCRTPoly> &matrixM, int is, int s, int d) {
        auto rotsM = this->getZeroCiphertext(d)->Clone();
        for (int j = 0; j < s; j++) {
            auto rotated_of_M = rot.rotate(matrixM, is * s * d + j * d);
            rotated_of_M->SetSlots(d*d*s);
            m_cc->EvalAddInPlace(
                rotsM, m_cc->EvalMult(rotated_of_M,
                                      m_cc->MakeCKKSPackedPlaintext(
                                          generateMaskVector(d*d*s, j, d), 1,
                                          0, nullptr, d*d*s)));
        }
        return rotsM;
    }


    Ciphertext<DCRTPoly> getZeroCiphertext(int d) {
        std::vector<double> zeroVec(d*d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB, int s1, int B1, int ng1, int nb1, int np1, int d){
        auto matrixC = this->getZeroCiphertext(d)->Clone();
        Ciphertext<DCRTPoly> babyStepsOfA[nb1];
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfB;

        // nb rotations required
        for (int i = 0; i < nb1; i++) {
            babyStepsOfA[i] = rot.rotate(matrixA, i);
        }
        for (int i = 0; i < np1; i++) {
            auto t = rot.rotate(matrixB, i * d);
            t->SetSlots(d*d*s1);
            babyStepsOfB.push_back(t);
        }

        for (int i = 0; i < B1; i++) {
            auto batched_rotations_B = vecRotsOpt(babyStepsOfB, i, s1, np1, d);
            auto diagA = this->getZeroCiphertext(d)->Clone();
            for (int k = -ng1; k < ng1; k++) {
                if (k < 0) {
                    auto tmp = this->getZeroCiphertext(d)->Clone();
                    auto babyStep = (k == -ng1) ? 1 : 0;

                    for (int j = d * d + k * nb1 + 1 + babyStep;
                         j <= d * d + (k + 1) * nb1; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s1, i * s1, j, d), -k * nb1);
                        m_cc->EvalAddInPlace(
                            tmp, m_cc->EvalMult(babyStepsOfA[babyStep],
                                                m_cc->MakeCKKSPackedPlaintext(
                                                    rotated_plain_vec, 1, 0,
                                                    nullptr, s1 * d * d)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb1));
                } else { // k>=0
                    auto tmp = this->getZeroCiphertext(d)->Clone();
                    auto babyStep = 0;
                    for (int j = k * nb1 + 1; j <= (k + 1) * nb1; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s1, i * s1, j, d), -k * nb1);
                        m_cc->EvalAddInPlace(
                            tmp, m_cc->EvalMult(babyStepsOfA[babyStep],
                                                m_cc->MakeCKKSPackedPlaintext(
                                                    rotated_plain_vec, 1, 0,
                                                    nullptr, d*d*s1)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb1));
                }
            }
            m_cc->EvalAddInPlace(matrixC,
                                 m_cc->EvalMult(diagA, batched_rotations_B));
        }
        for (int i = 1; i <= log2(s1); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                 rot.rotate(matrixC, (d*d*s1) / (1 << i)));
        }
        matrixC->SetSlots(d * d);

        return matrixC;
    }

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M, int s, int B, int ng, int nb, int np, int d, int r) {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d*d);

        auto trace = this->eval_trace(M, d, d*d);
        // upperBound = SAMPLE_DIM * FEATURE_DIM = 64 * 8 = 512
        double traceUpperBound = static_cast<double>(SAMPLE_DIM) * FEATURE_DIM;
        auto trace_reciprocal = this->eval_scalar_inverse(trace, traceUpperBound, 1, d*d);


        auto Y = this->m_cc->EvalMult(pI, trace_reciprocal);
        auto A_bar = this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(M, trace_reciprocal));

        for (int i = 0; i < r - 1; i++) {

            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
            A_bar = this->eval_mult(A_bar, A_bar, s, B, ng, nb, np, d);
            if ((int)Y->GetLevel() >= this->m_multDepth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
        if ((int)Y->GetLevel() >= this->m_multDepth - 2) {
                Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        return Y;
    }

public:
    LinearRegression_NewCol(std::shared_ptr<Encryption> enc,
                           CryptoContext<DCRTPoly> cc,
                           KeyPair<DCRTPoly> keyPair,
                           std::vector<int> rotIndices,
                           int multDepth)
        : LinearRegressionBase(enc, cc, keyPair, rotIndices)
        , m_maxBatch(cc->GetRingDimension() / 2)
        , m_multDepth(multDepth)
    {}

    TimingResult trainWithTimings(const Ciphertext<DCRTPoly>& X,
                                 const Ciphertext<DCRTPoly>& y) override {
        using namespace std::chrono;

        int SAMPLE_DIM = 64;
        int FEATURE_DIM = 8;

        if (m_verbose) {
            std::cout << "\n========== NewCol Encrypted LR Training ==========" << std::endl;
            debugPrintMatrix("Input X", X, SAMPLE_DIM, SAMPLE_DIM, SAMPLE_DIM);
        }

        // Step 1: X^tX using JKLS18 (unified 64x64 multiplication)
        if (m_verbose) std::cout << "[Step 1] Computing X^T * X with JKLS18..." << std::endl;
        auto step1_start = high_resolution_clock::now();
        auto Xt = eval_transpose(X, SAMPLE_DIM, SAMPLE_DIM * SAMPLE_DIM);
        auto XtX = eval_mult_JKLS18(Xt, X, SAMPLE_DIM);

        if (m_verbose) {
            debugPrintMatrix("X^T * X (64x64)", XtX, SAMPLE_DIM, SAMPLE_DIM, SAMPLE_DIM);
        }

        // re-batch
        if (m_verbose) std::cout << "[Step 1b] Rebatching X^T*X from 64x64 to 8x8..." << std::endl;
        auto rebatched_XtX = XtX->Clone();
        for(int i = 0; i < FEATURE_DIM-1; i++) {
            m_cc->EvalAddInPlace(rebatched_XtX,
                rot.rotate(XtX, (SAMPLE_DIM - FEATURE_DIM)*(i+1)));
        }
        std::vector<double> msk(FEATURE_DIM*FEATURE_DIM, 1.0);
        rebatched_XtX = m_cc->EvalMult(rebatched_XtX,
            m_cc->MakeCKKSPackedPlaintext(msk));

        for(int i = 0; i < log2((SAMPLE_DIM*SAMPLE_DIM)/(FEATURE_DIM*FEATURE_DIM)); i++) {
            m_cc->EvalAddInPlace(rebatched_XtX,
                rot.rotate(rebatched_XtX, -SAMPLE_DIM*(1<<i)));
        }
        rebatched_XtX->SetSlots(FEATURE_DIM*FEATURE_DIM);
        auto step1_end = high_resolution_clock::now();

        if (m_verbose) {
            debugPrintMatrix("X^T * X (rebatched 8x8)", rebatched_XtX, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
            double tr = debugGetTrace(rebatched_XtX, FEATURE_DIM);
            std::cout << "trace(X^T * X) = " << tr << std::endl;
        }

        // Step 2: Matrix inverse
        if (m_verbose) std::cout << "\n[Step 2] Computing (X^T * X)^{-1}..." << std::endl;
        int s2 = std::min(FEATURE_DIM, m_maxBatch / FEATURE_DIM /FEATURE_DIM);
        int B2 = FEATURE_DIM / s2;
        int ng2 = 2;
        int nb2 = 4;
        int np2 = 2;
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = eval_inverse(rebatched_XtX, s2, B2, ng2, nb2, np2, FEATURE_DIM, 18);
        auto step2_end = high_resolution_clock::now();

        if (m_verbose) {
            debugPrintMatrix("(X^T * X)^{-1}", inv_XtX, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        // Step 3: X^ty
        if (m_verbose) std::cout << "[Step 3] Computing X^T * y..." << std::endl;
        auto step3_start = high_resolution_clock::now();
        auto Xty = computeXty(X, y, FEATURE_DIM, SAMPLE_DIM);
        auto step3_end = high_resolution_clock::now();

        if (m_verbose) {
            debugPrintMatrix("X^T * y (8x8 view)", Xty, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        if (m_verbose) std::cout << "[Step 4] Computing weights..." << std::endl;
        auto step4_start = high_resolution_clock::now();

        if (m_verbose) {
            std::cout << "  inv_XtX slots: " << inv_XtX->GetSlots() << std::endl;
            std::cout << "  Xty slots: " << Xty->GetSlots() << std::endl;
        }

        auto Xty_transposed = eval_transpose(Xty, FEATURE_DIM, FEATURE_DIM * FEATURE_DIM);

        if (m_verbose) {
            std::cout << "  Xty_transposed slots: " << Xty_transposed->GetSlots() << std::endl;
            debugPrintMatrix("Xty_transposed", Xty_transposed, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        auto result = m_cc->EvalMultAndRelinearize(inv_XtX, Xty_transposed);

        if (m_verbose) {
            std::cout << "  result (after mult) slots: " << result->GetSlots() << std::endl;
            debugPrintMatrix("inv_XtX * Xty_transposed", result, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
            std::cout << "  [Folding sum] Starting with slots: " << result->GetSlots() << std::endl;

            Plaintext ptx;
            m_cc->Decrypt(m_keyPair.secretKey, result, &ptx);
            ptx->SetLength(256);
            auto vals = ptx->GetRealPackedValue();
            std::cout << "  Total vals size: " << vals.size() << std::endl;
            std::cout << "  Slots 64-71: ";
            for (int i = 64; i < 72 && i < (int)vals.size(); i++) {
                std::cout << std::setprecision(6) << vals[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "  Slots 128-135: ";
            for (int i = 128; i < 136 && i < (int)vals.size(); i++) {
                std::cout << std::setprecision(6) << vals[i] << " ";
            }
            std::cout << std::endl;
        }

        for (int i = (int)log2(FEATURE_DIM) - 1; i >= 0; i--) {
            int shift = FEATURE_DIM * (1 << i);
            m_cc->EvalAddInPlace(result, rot.rotate(result, shift));
            if (m_verbose) {
                std::cout << "  After rotate by " << shift << ":" << std::endl;
                debugPrintMatrix("result", result, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
            }
        }

        m_weights = result;
        auto step4_end = high_resolution_clock::now();

        if (m_verbose) {
            debugPrintMatrix("Weights (8x8)", m_weights, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }
};