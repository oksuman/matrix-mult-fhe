// lr_ar24.h
#pragma once

#include "lr_base.h"

class LinearRegression_AR24 : public LinearRegressionBase {
private:
    const int m_maxBatch;
    const int m_multDepth;

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
        int num_slots = d*d*s;

        auto matrixC = getZeroCiphertext(num_slots)->Clone();
        auto matrixA = matA->Clone();
        auto matrixB = matB->Clone();
        matA->SetSlots(d*d*s);
        matB->SetSlots(d*d*s);

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
            auto phi_si = m_cc->MakeCKKSPackedPlaintext(generatePhiMsk(s * i, d, s), 1, 0, nullptr, num_slots);
            auto tmp = m_cc->EvalMult(matrixA, phi_si);
            tmp = rot.rotate(tmp, s * i);
            for (int j = 0; j < log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j)));
            }
            Tilde_A[i] = tmp;
        }

        for (int i = 0; i < B; i++) {
            auto psi_si = m_cc->MakeCKKSPackedPlaintext(generatePsiMsk(s * i, d, s), 1, 0, nullptr, num_slots);
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
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, 350, 355, 5);
 
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

    Ciphertext<DCRTPoly> getZeroCiphertext(int batchSize) {
        std::vector<double> zeroVec(batchSize, 0.0);
        auto zeroPtx = this->m_cc->MakeCKKSPackedPlaintext(zeroVec, 1, 0, nullptr, batchSize);
        return this->m_cc->Encrypt(zeroPtx, this->m_keyPair.publicKey);
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
        

     
        int s1 = std::min(SAMPLE_DIM, m_maxBatch / SAMPLE_DIM /SAMPLE_DIM);

        auto step1_start = high_resolution_clock::now();
        auto Xt = eval_transpose(X, SAMPLE_DIM, 1<<16);
        auto XtX = eval_mult(Xt, X, SAMPLE_DIM, s1);
        XtX->SetSlots(SAMPLE_DIM*SAMPLE_DIM);

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

        int s2 = std::min(FEATURE_DIM, m_maxBatch / FEATURE_DIM /FEATURE_DIM);
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = eval_inverse(rebatched_XtX, s2, FEATURE_DIM, 18);
        auto step2_end = high_resolution_clock::now();

        auto step3_start = high_resolution_clock::now();
        for(int i=0; i<std::log2(s1); i++){
            this->m_cc->EvalAddInPlace(Xt, rot.rotate(Xt, -(SAMPLE_DIM*SAMPLE_DIM)*(1<<i)));
        }
        Xt->SetSlots(SAMPLE_DIM * SAMPLE_DIM);
        auto Xty = computeXty(Xt, y, FEATURE_DIM, SAMPLE_DIM);
        auto step3_end = high_resolution_clock::now();

        // Step 4: Final weight computation
        auto step4_start = high_resolution_clock::now();
        auto res = m_cc->EvalMult(inv_XtX, Xty);
        res->SetSlots(FEATURE_DIM * FEATURE_DIM);
        m_weights = res->Clone();

        for(int i=1; i<FEATURE_DIM; i++){
            std::vector<double> column_sum_msk;
            for(int j=0; j<FEATURE_DIM; j++){
                for(int k=0; k<FEATURE_DIM; k++){
                    if(k<FEATURE_DIM-i)
                        column_sum_msk.push_back(1);
                    else
                        column_sum_msk.push_back(0);
                }    
            }
            m_cc->EvalAddInPlace(m_weights, m_cc->EvalMult(rot.rotate(res, i), m_cc->MakeCKKSPackedPlaintext(column_sum_msk, 1, 0, nullptr, FEATURE_DIM*FEATURE_DIM))); 
        }

        for(int i=1; i<FEATURE_DIM; i++){
            std::vector<double> column_sum_msk;
            for(int j=0; j<FEATURE_DIM; j++){
                for(int k=0; k<FEATURE_DIM; k++){
                    if(k<i)
                        column_sum_msk.push_back(0);
                    else
                        column_sum_msk.push_back(1);
                }    
            }
            m_cc->EvalAddInPlace(m_weights,  m_cc->EvalMult(rot.rotate(res, -i), m_cc->MakeCKKSPackedPlaintext(column_sum_msk, 1, 0, nullptr, FEATURE_DIM*FEATURE_DIM)));
        }
        auto step4_end = high_resolution_clock::now();

        
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
        for (int i = 0; i < 64; i++) {
            std::cout << std::setprecision(6) << std::fixed 
                   << result_vec[i] << " ";
        }
        outFile << "\n";
    }
};