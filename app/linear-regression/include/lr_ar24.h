// lr_ar24.h
#pragma once

#include "lr_base.h"

class LinearRegression_AR24 : public LinearRegressionBase {
private:
    const int m_maxBatch;
    const int m_multDepth;

    std::vector<double> generatePhiMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = k; i < d * d; i += d) {
            msk[i] = 1;
        }
        return msk;
    }

    std::vector<double> generatePsiMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int j = k; j < k + d; j++) {
            msk[j] = 1;
        }
        return msk;
    }

    // AR24 matrix multiplication: d×d matrices with s copies
    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly>& matA,
                                  const Ciphertext<DCRTPoly>& matB,
                                  int d, int s) {
        int B = d / s;
        int num_slots = d * d * s;

        auto matrixC = makeZero(num_slots);
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
            auto phi_si = m_cc->MakeCKKSPackedPlaintext(generatePhiMsk(s * i, d), 1, 0, nullptr, d * d);
            auto tmp = m_cc->EvalMult(matrixA_copy, phi_si);
            tmp = rot.rotate(tmp, s * i);
            for (int j = 0; j < (int)log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j)));
            }
            Tilde_A[i] = tmp;
        }

        // Build Tilde_B
        for (int i = 0; i < B; i++) {
            auto psi_si = m_cc->MakeCKKSPackedPlaintext(generatePsiMsk(s * i, d), 1, 0, nullptr, d * d);
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

    Ciphertext<DCRTPoly> clean(const Ciphertext<DCRTPoly> &M, int s, int d) {
        std::vector<double> msk(d * d * s, 0.0);
        for (int i = 0; i < d * d; i++) {
            msk[i] = 1.0;
        }
        auto pmsk =
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d * s);

        return m_cc->EvalMult(M, pmsk);
    }

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M, int s, int d, int r) {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        std::vector<double> vI2 = this->initializeIdentityMatrix2(d, d*d*s);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d*d);
        Plaintext pI2 = this->m_cc->MakeCKKSPackedPlaintext(vI2, 1, 0, nullptr, d*d*s);

        auto trace = this->eval_trace(M, d, d * d);
        // upperBound = SAMPLE_DIM * FEATURE_DIM = 64 * 8 = 512
        double traceUpperBound = static_cast<double>(SAMPLE_DIM) * FEATURE_DIM;
        auto trace_reciprocal = this->eval_scalar_inverse(trace, traceUpperBound, 1, d * d);
 
        auto Y = this->m_cc->EvalMult(pI, trace_reciprocal);
        auto A_bar = this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(M, trace_reciprocal));
     
        Y->SetSlots(d*d*s);
        A_bar->SetSlots(d*d*s);
        Y = this->clean(Y, s, d);
        A_bar = this->clean(A_bar, s, d);

        for (int i = 0; i < r - 1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI2, A_bar), d, s);
            A_bar = this->eval_mult(A_bar, A_bar, d, s);

            if ((int)Y->GetLevel() >= this->m_multDepth - 3) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);

                A_bar->SetSlots(d * d * s);
                A_bar = this->clean(A_bar, s, d);
                Y->SetSlots(d * d * s);
                Y = this->clean(Y, s, d);
            } else {
                A_bar->SetSlots(d * d * s);
                A_bar = this->clean(A_bar, s, d);
                Y->SetSlots(d * d * s);
                Y = this->clean(Y, s, d);
            }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI2, A_bar), d, s);
        Y->SetSlots(d * d);
        if ((int)Y->GetLevel() >= this->m_multDepth - 3) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }

        return Y;
    }

public:
    LinearRegression_AR24(std::shared_ptr<Encryption> enc,
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

        if (m_verbose) {
            std::cout << "\n========== AR24 Encrypted LR Training ==========" << std::endl;
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

        // Rebatch
        if (m_verbose) std::cout << "[Step 1b] Rebatching X^T*X from 64x64 to 8x8..." << std::endl;
        auto rebatched_XtX = XtX->Clone();
        for(int i = 0; i < FEATURE_DIM-1; i++) {
            m_cc->EvalAddInPlace(rebatched_XtX,
                rot.rotate(XtX, (SAMPLE_DIM - FEATURE_DIM)*(i+1)));
        }
        std::vector<double> msk(FEATURE_DIM*FEATURE_DIM, 1.0);
        rebatched_XtX = m_cc->EvalMult(rebatched_XtX,
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, SAMPLE_DIM * SAMPLE_DIM));

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
        if (m_verbose) std::cout << "\n[Step 2] Computing (X^T * X)^{-1} with AR24..." << std::endl;
        int s2 = std::min(FEATURE_DIM, m_maxBatch / FEATURE_DIM /FEATURE_DIM);
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = eval_inverse(rebatched_XtX, s2, FEATURE_DIM, 18);
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
            debugPrintVector("X^T * y", Xty, FEATURE_DIM);
        }

        // Step 4: Final weight computation (unified with newcol method)
        if (m_verbose) std::cout << "[Step 4] Computing weights..." << std::endl;
        auto step4_start = high_resolution_clock::now();

        auto Xty_transposed = eval_transpose(Xty, FEATURE_DIM, FEATURE_DIM * FEATURE_DIM);

        if (m_verbose) {
            debugPrintMatrix("Xty_transposed", Xty_transposed, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        auto result = m_cc->EvalMultAndRelinearize(inv_XtX, Xty_transposed);

        if (m_verbose) {
            debugPrintMatrix("inv_XtX * Xty_transposed", result, FEATURE_DIM, FEATURE_DIM, FEATURE_DIM);
        }

        for (int i = (int)log2(FEATURE_DIM) - 1; i >= 0; i--) {
            int shift = FEATURE_DIM * (1 << i);
            m_cc->EvalAddInPlace(result, rot.rotate(result, shift));
        }

        m_weights = result;
        auto step4_end = high_resolution_clock::now();

        if (m_verbose) {
            debugPrintVector("Weights", m_weights, FEATURE_DIM);
        }

        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }
};