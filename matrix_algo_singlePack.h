
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "encryption.h"
#include <memory>
#include <openfhe.h>
#include <vector>

using namespace lbcrypto;

template <int d>
class MatrixOperationBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    const Ciphertext<DCRTPoly> m_zeroCache;

    virtual Ciphertext<DCRTPoly> createZeroCache() {
        std::vector<double> zeroVec(d * d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

public:
    MatrixOperationBase(std::shared_ptr<Encryption> enc,
                       CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey)
        : m_enc(enc), m_cc(cc), m_PublicKey(publicKey),
         m_zeroCache(createZeroCache()) {}

    virtual ~MatrixOperationBase() = default;

    virtual Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
             const Ciphertext<DCRTPoly> &matrixB) = 0;

    std::vector<double> vectorRotate(const std::vector<double> &vec,
                                   int rotateIndex) {
        if (vec.empty())
            return std::vector<double>();
            
        std::vector<double> result = vec;
        int n = result.size();
        
        if (rotateIndex > 0) {
            std::rotate(result.begin(), result.begin() + rotateIndex,
                      result.end());
        }
        else if (rotateIndex < 0) {
            rotateIndex += n;
            std::rotate(result.begin(), result.begin() + rotateIndex,
                      result.end());
        }
        return result;
    }

    std::vector<double> generateTransposeMsk(int k) {
        std::set<int> indices;
        std::vector<double> msk(d*d, 0);
        
        if (k >= 0) {
            for (int j = 0; j < d - k; j++) {
                indices.insert((d + 1) * j + k);
            }
        } else {
            for (int j = -k; j < d; j++) {
                indices.insert((d + 1) * j + k);
            }
        }
        
        for (int i = 0; i < d * d; i++) {
            if (indices.find(i) != indices.end()) {
                msk[i] = 1.0;
            }
        }
        
        return msk;
    }

    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M) {
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfM(8);
        
        #pragma omp parallel for
        for(int i = 0; i < 8; i++) {
            babyStepsOfM[i] = m_cc->EvalRotate(M, (d-1)*i);
        }
        
        auto M_transposed = this->getZero()->Clone();
        
        for(int i = -8; i < 8; i++) {
            auto tmp = this->getZero()->Clone();
            const int js = (i == -8) ? 1 : 0;
            
            #pragma omp parallel for
            for(int j = js; j < 8; j++) {
                auto vmsk = generateTransposeMsk(8*i+j);
                vmsk = this->vectorRotate(vmsk, -8*(d-1)*i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(vmsk);
                auto masked = m_cc->EvalMult(babyStepsOfM[j], pmsk);
                
                #pragma omp critical
                {
                    m_cc->EvalAddInPlace(tmp, masked);
                }
            }
            
            tmp = m_cc->EvalRotate(tmp, 8*(d-1)*i);
            m_cc->EvalAddInPlace(M_transposed, tmp);
        }
        
        return M_transposed;
    }
 
    Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> M, int batchSize) {
        std::vector<double> msk(batchSize, 0);
        
        #pragma omp parallel for
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }

        auto trace = m_cc->EvalMult(M, m_cc->MakeCKKSPackedPlaintext(msk));
        
        for (int i = 1; i <= log2(batchSize); i++) {
            auto rotated = m_cc->EvalRotate(trace, batchSize / (1 << i));
            m_cc->EvalAddInPlace(trace, rotated);
        }
        
        return trace;
    }

    virtual const Ciphertext<DCRTPoly> &getZero() const { return m_zeroCache; }
    constexpr size_t getMatrixSize() const { return d; }
};

// Our proposed method, column-based approach, optimized for 64 * 64 matrix
template <int d> class MatrixMult_newColOpt : public MatrixOperationBase<d> {
  private:
    static constexpr int num_slots = 1<<16;
    static constexpr int s = 16;
    static constexpr int B = 4;

    static constexpr int ng = 2; // giant-step
    static constexpr int nb = 32; // baby-step
    static constexpr int np = 8; // precomptutation for VecRots
  protected:
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixMult_newColOpt(std::shared_ptr<Encryption> enc,
                      CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly>
                      publicKey)
        : MatrixOperationBase<d>(enc, cc, publicKey) 
    {}


    std::vector<double> generateMaskVector(int k) {
        std::vector<double> result(num_slots, 0.0);
        #pragma omp parallel for
        for (int i = k * d * d; i < (k + 1) * d * d; ++i) {
            result[i] = 1.0;
        }
        return result;
    }

    std::vector<double> genDiagVector(int k, int diag_index) {
        std::vector<double> result(d * d, 0.0);
        if (diag_index < 1 || diag_index > d * d ||
            (diag_index > d && diag_index < d * d - (d - 1))) {
            return result;
        }

        #pragma omp parallel for
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
            #pragma omp parallel for collapse(2)
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
        #pragma omp parallel for
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

    std::vector<double> genBatchDiagVector(int s, int k, int diag_index) {
        std::vector<double> result;
        result.reserve(d * d * s);

        for (int i = 0; i < s; ++i) {
            std::vector<double> diag_vector = genDiagVector(k + i,
            diag_index); result.insert(result.end(), diag_vector.begin(),
            diag_vector.end());
        }

        return result;
    }

    Ciphertext<DCRTPoly> vecRotsOpt(const std::vector<Ciphertext<DCRTPoly>>&matrixM, int is){
        auto rotsM = this->getZero()->Clone();

        for (int j = 0; j < s / np; j++) {

            auto T = this->getZero()->Clone();

            for(int i=0; i<np; i++){
                auto msk = generateMaskVector(np * j + i);
                msk = vectorRotate(msk, -is * d * s - j * d * np);

                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr,num_slots);
                m_cc->EvalAddInPlace(T, m_cc->EvalMult(matrixM[i], pmsk));
            }
            m_cc->EvalAddInPlace(rotsM, m_cc->EvalRotate(T, is * d * s + j * d * np));
        }

        return rotsM;
    }

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {

        auto matrixC = this->getZero()->Clone();
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfA(nb);
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfB(np);
       
       
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < nb; i++) {
                babyStepsOfA[i] = m_cc->EvalRotate(matrixA, i);
            }
            
            #pragma omp for
            for (int i = 0; i < np; i++) {
                Ciphertext<DCRTPoly> t;
                t = m_cc->EvalRotate(matrixB, i*d);
                t->SetSlots(num_slots);
                babyStepsOfB[i] = t;
            }
        }


        for (int i = 0; i < B; i++) {
            auto batched_rotations_B = vecRotsOpt(babyStepsOfB, i);
            auto diagA = this->getZero()->Clone();
            
            for (int k = -ng; k < ng; k++) {
                auto tmp = this->getZero()->Clone();
                
                if (k < 0) {
                    const int startIdx = d * d + k * nb + 1;
                    const int endIdx = d * d + (k + 1) * nb;
                    
                    #pragma omp parallel for
                    for (int j = startIdx; j <= endIdx; j++) {
                        // Calculate babyStep index directly from j
                        int localBabyStep = (k == -ng && j == startIdx) ? 0 : (j - startIdx);
                        
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                            
                        auto mult_result = m_cc->EvalMult(
                            babyStepsOfA[localBabyStep],
                            m_cc->MakeCKKSPackedPlaintext(
                                rotated_plain_vec, 1, 0, nullptr, num_slots));
                                
                        #pragma omp critical
                        {
                            m_cc->EvalAddInPlace(tmp, mult_result);
                        }
                    }
                } else {
                    const int startIdx = k * nb + 1;
                    const int endIdx = (k + 1) * nb;
                    
                    #pragma omp parallel for
                    for (int j = startIdx; j <= endIdx; j++) {
                        // Calculate babyStep index directly from j
                        int localBabyStep = j - startIdx;
                        
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                            
                        auto mult_result = m_cc->EvalMult(
                            babyStepsOfA[localBabyStep],
                            m_cc->MakeCKKSPackedPlaintext(
                                rotated_plain_vec, 1, 0, nullptr, num_slots));
                                
                        #pragma omp critical
                        {
                            m_cc->EvalAddInPlace(tmp, mult_result);
                        }
                    }
                }
                
                m_cc->EvalAddInPlace(diagA, m_cc->EvalRotate(tmp, k * nb));
            }
            
            m_cc->EvalAddInPlace(matrixC, m_cc->EvalMult(diagA, batched_rotations_B));
        }

        for (int i = 1; i <= log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                 m_cc->EvalRotate(matrixC, num_slots / (1 << i)));
        }
        matrixC->SetSlots(d * d);

        return matrixC;
    }
};