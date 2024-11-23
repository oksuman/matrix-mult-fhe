// lr_as24.h
#pragma once

#include "lr_base.h"

class LinearRegression_AS24 : public LinearRegressionBase {
private:
    const int m_maxBatch;

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

    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly>& matA,
                                  const Ciphertext<DCRTPoly>& matB, 
                                  int d, int s) {
        int B = d / s;
        auto matrixC = getZeroCiphertext(d)->Clone();
        auto matrixA = matA->Clone();
        auto matrixB = matB->Clone();

        std::vector<Ciphertext<DCRTPoly>> Tilde_A(B);
        std::vector<Ciphertext<DCRTPoly>> Tilde_B(B);

        for (int i = 0; i < log2(s); i++) {
            auto tmp = rot.rotate(matrixA, (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixA, tmp);
        }
        for (int i = 0; i < log2(s); i++) {
            auto tmp = rot.rotate(matrixB, d * (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixB, tmp);
        }

        for (int i = 0; i < B; i++) {
            auto phi_si = m_cc->MakeCKKSPackedPlaintext(generatePhiMsk(s * i, d, s));
            auto tmp = m_cc->EvalMult(matrixA, phi_si);
            tmp = rot.rotate(tmp, s * i);
            for (int j = 0; j < log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j)));
            }
            Tilde_A[i] = tmp;
        }

        for (int i = 0; i < B; i++) {
            auto psi_si = m_cc->MakeCKKSPackedPlaintext(generatePsiMsk(s * i, d, s));
            auto tmp = m_cc->EvalMult(matrixB, psi_si);
            tmp = rot.rotate(tmp, s * i * d);
            for (int j = 0; j < log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j) * d));
            }
            Tilde_B[i] = tmp;
        }

        for (int i = 0; i < B; i++) {
            m_cc->EvalAddInPlace(matrixC, 
                m_cc->EvalMultAndRelinearize(Tilde_A[i], Tilde_B[i]));
        }

        for (int i = 0; i < log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC, rot.rotate(matrixC, (d * d) * (1 << i)));
        }
        matrixC->SetSlots(d * d);
        return matrixC;
    }

    Ciphertext<DCRTPoly> getZeroCiphertext(int d) {
        std::vector<double> zeroVec(d * d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

public:
    LinearRegression_AS24(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         KeyPair<DCRTPoly> keyPair,
                         std::vector<int> rotIndices)
        : LinearRegressionBase(enc, cc, keyPair, rotIndices)
        , m_maxBatch(cc->GetRingDimension() / 2)
    {}

    TimingResult trainWithTimings(const Ciphertext<DCRTPoly>& X,
                                 const Ciphertext<DCRTPoly>& y) override {
        using namespace std::chrono;
        
        std::ofstream logFile("intermediate_results.txt");
        logFile << "=== Linear Regression Intermediate Results ===\n";
     
        int s1 = std::min(SAMPLE_DIM, m_maxBatch / SAMPLE_DIM /SAMPLE_DIM);

        auto step1_start = high_resolution_clock::now();
        auto Xt = eval_transpose(X, SAMPLE_DIM);
        logIntermediateResult("Xt", Xt, logFile);
        auto XtX = eval_mult(Xt, X, SAMPLE_DIM, s1);
        XtX->SetSlots(SAMPLE_DIM*SAMPLE_DIM);
        logIntermediateResult("XtX", XtX, logFile);

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
        logIntermediateResult("Rebatched XtX", rebatched_XtX, logFile);
        auto step1_end = high_resolution_clock::now();

        int s2 = std::min(FEATURE_DIM, m_maxBatch / FEATURE_DIM /FEATURE_DIM);
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = eval_inverse(rebatched_XtX, FEATURE_DIM, s2, 20);
        logIntermediateResult("Inverse of XtX", inv_XtX, logFile);
        auto step2_end = high_resolution_clock::now();

        auto step3_start = high_resolution_clock::now();
        auto Xty = computeXty(Xt, y, FEATURE_DIM, SAMPLE_DIM);
        logIntermediateResult("Final Xty", Xty, logFile);
        auto step3_end = high_resolution_clock::now();

        auto step4_start = high_resolution_clock::now();
        m_weights = eval_mult(inv_XtX, Xty, FEATURE_DIM, s2);
        logIntermediateResult("Final Weights", m_weights, logFile);
        auto step4_end = high_resolution_clock::now();

        logFile.close();
        
        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }

    void logIntermediateResult(const std::string& label, 
                              const Ciphertext<DCRTPoly>& cipher,
                              std::ofstream& outFile) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result_vec = ptx->GetRealPackedValue();
        
        outFile << "\n=== " << label << " ===\n";
        std::cout << "\n=== " << label << " ===\n";
        outFile << "Number of slots: " << cipher->GetSlots() << "\n";
        outFile << "First 10 elements: \n";
        for (int i = 0; i < std::min(10, (int)result_vec.size()); i++) {
            outFile << std::setprecision(6) << std::fixed 
                   << result_vec[i] << " ";
            std::cout << std::setprecision(6) << std::fixed 
                   << result_vec[i] << " ";
        }
        outFile << "\n";
    }
};