#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "encryption.h"
#include "rotation.h"
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
    std::set<int> m_rotations; 


    virtual Ciphertext<DCRTPoly> createZeroCache() {
        std::vector<double> zeroVec(d * d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

public:
    MatrixOperationBase(std::shared_ptr<Encryption> enc,
                       CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey,
                       std::vector<int> rotations)
        : m_enc(enc), m_cc(cc), m_PublicKey(publicKey),
        m_zeroCache(createZeroCache()) {
            for(const auto& index : rotations){
                m_rotations.insert(index);
            }
        }

    virtual ~MatrixOperationBase() = default;

    virtual Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
             const Ciphertext<DCRTPoly> &matrixB) = 0;

    Ciphertext<DCRTPoly> rotateCtx(Ciphertext<DCRTPoly> cin, int index) {
        if (index == 0)
            return cin;

        if (m_rotations.count(index)) {
            return m_cc->EvalRotate(cin, index);
        } else {
            Ciphertext<DCRTPoly> result = cin;
            int bitPos = 0;
            int absIndex = std::abs(index);
            
            while (absIndex > 0) {
                if (absIndex & 1) {
                    int powerOf2 = (1 << bitPos);
                    int rotIndex = powerOf2;
                    
                    if (!m_rotations.count(rotIndex)) {
                        throw std::runtime_error("Required power-of-2 rotation not available: " + std::to_string(rotIndex));
                    }
                    
                    result = m_cc->EvalRotate(result, rotIndex);
                }
                
                absIndex >>= 1;
                bitPos++;
            }
            
            return result;
        }
    }

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
        
        // #pragma omp parallel for // need to check the performance
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
            babyStepsOfM[i] = rotateCtx(M, (d-1)*i);
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
            
            tmp = rotateCtx(tmp, 8*(d-1)*i);
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
            auto rotated = rotateCtx(trace, batchSize / (1 << i));
            m_cc->EvalAddInPlace(trace, rotated);
        }
        
        return trace;
    }

    virtual const Ciphertext<DCRTPoly> &getZero() const { return m_zeroCache; }
    constexpr size_t getMatrixSize() const { return d; }
};

// For test
template <int d> class TestMatrixOperation : public MatrixOperationBase<d> {
  protected:
    using MatrixOperationBase<d>::m_cc;

  public:
    using MatrixOperationBase<d>::MatrixOperationBase;

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {
        return nullptr;
    }
};

// optimized for 64 by 64 matrix inversion 
template <int d>
class MatrixMult_AS24Opt : public MatrixOperationBase<d> {
private:
    static constexpr int B = 4;
    static constexpr int s = 16;

protected:
    static constexpr int max_batch = 1 << 16;
    using MatrixOperationBase<d>::m_cc;

public:
    MatrixMult_AS24Opt(std::shared_ptr<Encryption> enc, CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey, std::vector<int> rotations)
        : MatrixOperationBase<d>(enc, cc, publicKey, rotations) {}

    std::vector<double> generatePhiMsk(int k) {
        std::vector<double> msk(max_batch, 0);
        
        #pragma omp parallel for schedule(static)
        for (int i = k; i < max_batch; i += d) {
            msk[i] = 1;
        }
        return msk;
    }

    std::vector<double> generatePsiMsk(int k) {
        std::vector<double> msk(max_batch, 0);
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < s; i++) {
            for (int j = i * d * d + k * d; j < i * d * d + k * d + d; j++) {
                msk[j] = 1;
            }
        }
        return msk;
    }

    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly> &matA,
                                  const Ciphertext<DCRTPoly> &matB) override {
        auto matrixC = this->getZero()->Clone();
        auto matrixA = matA->Clone();
        auto matrixB = matB->Clone();
        std::vector<Ciphertext<DCRTPoly>> Tilde_A(B);
        std::vector<Ciphertext<DCRTPoly>> Tilde_B(B);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                for (int i = 0; i < log2(s); i++) {
                    auto tmp = this->rotateCtx(matrixA, (1 << i) - d * d * (1 << i));
                    m_cc->EvalAddInPlace(matrixA, tmp);
                }
            }
            
            #pragma omp section
            {
                for (int i = 0; i < log2(s); i++) {
                    auto tmp = this->rotateCtx(matrixB, d * (1 << i) - d * d * (1 << i));
                    m_cc->EvalAddInPlace(matrixB, tmp);
                }
            }
        }

        // Targeting 64 by 64 matrix
        // Note: B=4, s=16
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < B; i++) {
                auto phi_si = m_cc->MakeCKKSPackedPlaintext(
                    generatePhiMsk(s * i), 1, 0, nullptr, max_batch);
                auto tmp = m_cc->EvalMult(matrixA, phi_si);
                switch (i)
                {
                case 0:
                    for (int j = 0; j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 1:
                    for (int j = 0; j < log2(s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(s); j < log2(2*s); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(2*s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 2:
                    for (int j = 0; j < log2(2*s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(2*s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 3:
                    for (int j = 0; j < log2(s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j));
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                default:
                    break;
                }
                Tilde_A[i] = tmp;
            }

            #pragma omp for nowait
            for (int i = 0; i < B; i++) {
                auto psi_si = m_cc->MakeCKKSPackedPlaintext(
                    generatePsiMsk(s * i), 1, 0, nullptr, max_batch);
                auto tmp = m_cc->EvalMult(matrixB, psi_si);
                switch (i)
                {
                case 0:
                    for (int j = 0; j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 1:
                    for (int j = 0; j < log2(s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(s); j < log2(2*s); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(2*s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 2:
                    for (int j = 0; j < log2(2*s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(2*s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                case 3:
                    for (int j = 0; j < log2(s); j++) {
                        auto rotated = this->rotateCtx(tmp, -(1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    for (int j = log2(s); j < log2(d); j++) {
                        auto rotated = this->rotateCtx(tmp, (1 << j) * d);
                        m_cc->EvalAddInPlace(tmp, rotated);
                    }
                    break;
                default:
                    break;
                }
                Tilde_B[i] = tmp;
            }
        }
       
        #pragma omp for
        for (int i = 0; i < B; i++) {
            auto mult = m_cc->EvalMultAndRelinearize(Tilde_A[i], Tilde_B[i]);
            #pragma omp critical
            m_cc->EvalAddInPlace(matrixC, mult);
        }

        for (int i = 0; i < log2(s); i++) {
            auto rotated = this->rotateCtx(matrixC, (d * d) * (1 << i));
            m_cc->EvalAddInPlace(matrixC, rotated);
        }

        matrixC->SetSlots(d * d);
        return matrixC;
    }

    Ciphertext<DCRTPoly> clean(const Ciphertext<DCRTPoly> &M) {
        std::vector<double> msk(max_batch, 0.0);
        
        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < d * d; i++) {
            msk[i] = 1.0;
        }
        
        auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, max_batch);
        return m_cc->EvalMult(M, pmsk);
    }

    static constexpr int getMaxBatch() { return max_batch; }
    static constexpr int getB() { return B; }
    static constexpr int getS() { return s; }
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
                      publicKey, std::vector<int> rotations)
        : MatrixOperationBase<d>(enc, cc, publicKey, rotations) 
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

    Ciphertext<DCRTPoly> vecRotsOpt(const std::vector<Ciphertext<DCRTPoly>>& matrixM, int is) {
        auto rotsM = this->getZero()->Clone();
        
        for (int j = 0; j < s / np; j++) {  
            std::vector<Ciphertext<DCRTPoly>> temp_results(np);
            
            #pragma omp parallel for num_threads(2)
            for (int i = 0; i < np; i++) {
                auto msk = generateMaskVector(np * j + i);
                msk = vectorRotate(msk, -is * d * s - j * d * np);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, num_slots);
                temp_results[i] = m_cc->EvalMult(matrixM[i], pmsk);
            }
            
            auto T = m_cc->EvalAddMany(temp_results);
            m_cc->EvalAddInPlace(rotsM, this->rotateCtx(T, is * d * s + j * d * np));
        }
        
        return rotsM;
    }

    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly> &matrixA,
                                const Ciphertext<DCRTPoly> &matrixB) override {
        auto matrixC = this->getZero()->Clone();
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfA(nb);
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfB(np);
        
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < nb; i++) {
                babyStepsOfA[i] = this->rotateCtx(matrixA, i);
            }
            
            #pragma omp for
            for (int i = 0; i < np; i++) {
                Ciphertext<DCRTPoly> t = this->rotateCtx(matrixB, i*d);
                t->SetSlots(num_slots);
                babyStepsOfB[i] = t;
            }
        }
        
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < B; i++) {
            auto batched_rotations_B = vecRotsOpt(babyStepsOfB, i);
            auto diagA = this->getZero()->Clone();
            
            #pragma omp parallel for num_threads(3)
            for (int k = -ng; k < ng; k++) {
                auto tmp = this->getZero()->Clone();
                
                if (k < 0) {
                    const int startIdx = d * d + k * nb + 1;
                    const int endIdx = d * d + (k + 1) * nb;
                    
                    for (int j = startIdx; j <= endIdx; j++) {
                        int localBabyStep = (k == -ng && j == startIdx) ? 0 : (j - startIdx);
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                        auto mult_result = m_cc->EvalMult(
                            babyStepsOfA[localBabyStep],
                            m_cc->MakeCKKSPackedPlaintext(
                                rotated_plain_vec, 1, 0, nullptr, num_slots));
                        m_cc->EvalAddInPlace(tmp, mult_result);
                    }
                } else {
                    const int startIdx = k * nb + 1;
                    const int endIdx = (k + 1) * nb;
                    
                    for (int j = startIdx; j <= endIdx; j++) {
                        int localBabyStep = j - startIdx;
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                        auto mult_result = m_cc->EvalMult(
                            babyStepsOfA[localBabyStep],
                            m_cc->MakeCKKSPackedPlaintext(
                                rotated_plain_vec, 1, 0, nullptr, num_slots));
                        m_cc->EvalAddInPlace(tmp, mult_result);
                    }
                }
                #pragma omp critical
                {
                    m_cc->EvalAddInPlace(diagA, this->rotateCtx(tmp, k * nb));
                }
            }
            
            auto result_B = m_cc->EvalMult(diagA, batched_rotations_B);
            #pragma omp critical
            {
                m_cc->EvalAddInPlace(matrixC, result_B);
            }
        }
        
        for (int i = 1; i <= log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                this->rotateCtx(matrixC, num_slots / (1 << i)));
        }
        
        matrixC->SetSlots(d * d);
        return matrixC;
    }
};