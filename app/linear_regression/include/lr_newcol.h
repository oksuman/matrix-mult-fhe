#pragma once

#include "lr_base.h"

class LinearRegression_NewCol : public LinearRegressionBase {
private:
    const int m_maxBatch;

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

    // Ciphertext<DCRTPoly>
    // vecRotsOpt(const std::vector<Ciphertext<DCRTPoly>> &matrixM, int is, int d) {
    //     auto rotsM = this->getZeroCiphertext(d)->Clone();
    //     for (int j = 0; j < m_s / m_np; j++) {

    //         auto T = this->getZeroCiphertext(d)->Clone();

    //         for (int i = 0; i < m_np; i++) {
    //             auto msk = generateMaskVector(m_numSlots, m_np * j + i, d);
    //             msk = vectorRotate(msk, -is * d * m_s - j * d * m_np);

    //             auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr,
    //                                                       m_numSlots);
    //             m_cc->EvalAddInPlace(T, m_cc->EvalMult(matrixM[i], pmsk));
    //         }
    //         m_cc->EvalAddInPlace(rotsM, rot.rotate(T, is * d * m_s + j * d * m_np));
    //     }

    //     return rotsM;
    // }

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
        std::ofstream logFile("intermediate_results.txt");

        auto matrixC = this->getZeroCiphertext(d)->Clone();
        Ciphertext<DCRTPoly> babyStepsOfA[nb1];
        // std::vector<Ciphertext<DCRTPoly>> babyStepsOfB;

        // nb rotations required
        for (int i = 0; i < nb1; i++) {
            babyStepsOfA[i] = rot.rotate(matrixA, i);
        }

        for (int i = 0; i < B1; i++) {
            auto batched_rotations_B = vecRots(matrixB, i, s1, d);
            // logIntermediateResult("batched_rotations_B", batched_rotations_B, logFile);
            
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
            // logIntermediateResult("diagA", diagA, logFile);

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
        std::ofstream logFile("intermediate_results.txt");
        logIntermediateResult("XtX in inverse", M, logFile);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d*d);

        std::cout << "start trace" << std::endl;

        auto trace = this->eval_trace(M, d, d*d);
        logIntermediateResult("trace", trace, logFile);

        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, 350, 355, 5);
        logIntermediateResult("1/trace", trace_reciprocal, logFile);
 

        auto Y = this->m_cc->EvalMult(pI, trace_reciprocal);
        auto A_bar = this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(M, trace_reciprocal));
    
        for (int i = 0; i < r - 1; i++) {
            // if (d >= 8 && (int)Y->GetLevel() >= this->depth - 2) {
            //     A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
            //     Y = m_cc->EvalBootstrap(Y, 2, 17);
            // }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
            A_bar = this->eval_mult(A_bar, A_bar, s, B, ng, nb, np, d);
            std::cout << "i: " << i << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar), s, B, ng, nb, np, d);
        return Y;
    }
    
    void logIntermediateResult(const std::string& label, 
                              const Ciphertext<DCRTPoly>& cipher,
                              std::ofstream& outFile) {
        Plaintext ptx;
        m_cc->Decrypt(this->m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result_vec = ptx->GetRealPackedValue();
        
        outFile << "\n=== " << label << " ===\n";
        std::cout << "\n=== " << label << " ===\n";
        outFile << "Number of slots: " << cipher->GetSlots() << "\n";
        outFile << "First 10 elements: \n";
        for (int i = 0; i < std::min(10, (int)result_vec.size()); i++) {
            outFile << std::setprecision(6) << std::fixed 
                   << result_vec[i] << " ";
        }
        for (int i = 0; i < std::min(10, (int)result_vec.size()); i++) {
            std::cout << std::setprecision(6) << std::fixed 
                   << result_vec[i] << " ";
        }
        std::cout << "\n";
        outFile << "\n";
    }

public:
    LinearRegression_NewCol(std::shared_ptr<Encryption> enc,
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
     
        int SAMPLE_DIM = 64;
        int FEATURE_DIM = 8;

        // Step 1: X^tX
        logFile << "\nAfter X multiplication:\n";
        logIntermediateResult("X", X, logFile);

        int s1 = std::min(SAMPLE_DIM, m_maxBatch / SAMPLE_DIM /SAMPLE_DIM);
        int B1 = SAMPLE_DIM / s1; 
        int ng1 = 1; 
        int nb1 = 64;
        int np1 = 0; 

        auto step1_start = high_resolution_clock::now();
        auto Xt = eval_transpose(X, SAMPLE_DIM);
        logFile << "\nAfter Xt multiplication:\n";
        logIntermediateResult("Xt", Xt, logFile);
        auto XtX = eval_mult(Xt, X, s1, B1, ng1, nb1, np1, SAMPLE_DIM);
        logFile << "\nAfter X^TX multiplication:\n";
        logIntermediateResult("XtX", XtX, logFile);

        // re-batch
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
        std::cout << "end" << std::endl;
        auto step1_end = high_resolution_clock::now();

        // Step 2: Matrix inverse
        int s2 = std::min(FEATURE_DIM, m_maxBatch / FEATURE_DIM /FEATURE_DIM);
        int B2 = FEATURE_DIM / s2; 
        int ng2 = 2; 
        int nb2 = 4;
        int np2 = 0; 
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = eval_inverse(rebatched_XtX, s2, B2, ng2, nb2, np2, FEATURE_DIM, 20);
        logIntermediateResult("Inverse of XtX", inv_XtX, logFile);
        auto step2_end = high_resolution_clock::now();

        // Step 3: X^ty
        auto step3_start = high_resolution_clock::now();
        auto Xty = computeXty(Xt, y, FEATURE_DIM, SAMPLE_DIM);
        logIntermediateResult("Final Xty", Xty, logFile);
        auto step3_end = high_resolution_clock::now();

        // Step 4: Final weight computation
        auto step4_start = high_resolution_clock::now();
        auto res = m_cc->EvalMult(inv_XtX, Xty);
        res->SetSlots(FEATURE_DIM * FEATURE_DIM);
        logIntermediateResult("res", res, logFile);
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
        // m_weights->SetSlots(FEATURE_DIM);
        logIntermediateResult("Final Weights", m_weights, logFile);
        auto step4_end = high_resolution_clock::now();
        std::cout << std::endl;
        logFile.close();
        
        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }
};