
#pragma once

#include "encryption.h"
#include "rotation.h"
#include <memory>
#include <openfhe.h>
#include <vector>

using namespace lbcrypto;

/*
 *   This code contains matrix algorithms
 *   where a matrix is encrypted into a single ciphertext.
 */

template <int d> // Matrix dimension d x d
class MatrixOperationBase {
  protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    RotationComposer rot;
    const Ciphertext<DCRTPoly> m_zeroCache;

    virtual Ciphertext<DCRTPoly> createZeroCache() {
        std::vector<double> zeroVec(d * d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

  public:
    MatrixOperationBase(std::shared_ptr<Encryption> enc,
                        CryptoContext<DCRTPoly> cc,
                        PublicKey<DCRTPoly> publicKey,
                        std::vector<int> rotIndices)
        : m_enc(enc), m_cc(cc), m_PublicKey(publicKey),
          rot(cc, rotIndices, cc->GetRingDimension() / 2),
          m_zeroCache(createZeroCache()) {}

    virtual ~MatrixOperationBase() = default;
    virtual Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) = 0;

    std::vector<double> vectorRotate(const std::vector<double> &vec,
                                     int rotateIndex) {
        if (vec.empty())
            return std::vector<double>();

        std::vector<double> result = vec; // 원본 벡터를 복사
        int n = result.size();

        if (rotateIndex > 0) // left rotation
            std::rotate(result.begin(), result.begin() + rotateIndex,
                        result.end());
        else if (rotateIndex < 0) { // right rotation
            rotateIndex += n;
            std::rotate(result.begin(), result.begin() + rotateIndex,
                        result.end());
        }
        return result;
    }

    std::vector<double> generateShiftingMsk(int k) {
        std::vector<double> v(d * d, 0);
        for (int i = k; i < d * d; i += d) {
            for (int j = i; j < i + d - k; ++j) {
                v[j] = 1;
            }
        }
        return v;
    }

    Ciphertext<lbcrypto::DCRTPoly> columnShifting(const Ciphertext<DCRTPoly> M,
                                                  int l) {
        Ciphertext<DCRTPoly> shiftResult;
        if (l == 0)
            return M;
        else {
            std::vector<double> msk = generateShiftingMsk(l);
            Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(msk);

            auto tmp = m_cc->EvalMult(pmsk, M);

            auto M_1 = rot.rotate(m_cc->EvalSub(M, tmp), l - d);
            auto M_2 = rot.rotate(tmp, l);

            return m_cc->EvalAdd(M_1, M_2);
        }
    }

    std::vector<double> generateTransposeMsk(int k) {
        std::set<int> indices;
        if (k >= 0) {
            for (int j = 0; j < d - k; j++) {
                indices.insert((d + 1) * j + k);
            }
        } else {
            for (int j = -k; j < d; j++) {
                indices.insert((d + 1) * j + k);
            }
        }
        std::vector<double> msk(d * d, 0);
        for (int index : indices) {
            msk[index] = 1.0;
        }
        return msk;
    }

    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M) {
        // initializatino (i=0)
        auto p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMsk(0));
        auto M_transposed = m_cc->EvalMult(M, p);

        // case i > 0
        for (int i = 1; i < d; i++) {
            auto p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMsk(i));
            m_cc->EvalAddInPlace(M_transposed,
                                 m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }
        // case i < 0
        for (int i = -1; i > -d; i--) {
            auto p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMsk(i));
            m_cc->EvalAddInPlace(M_transposed,
                                 m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }

        return M_transposed;
    }

    Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> M, int batchSize) {
        /*
         * batch size : d*d or d*d*d
         * Only RT22 uses d*d*d batch
         */
        std::vector<double> msk(batchSize, 0);
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }
        auto trace = m_cc->EvalMult(M, m_cc->MakeCKKSPackedPlaintext(msk));

        // Require EvalSum key
        // trace_M = m_cc->EvalSum(trace_M, d*d);
        for (int i = 1; i <= log2(batchSize); i++) {
            m_cc->EvalAddInPlace(trace,
                                 rot.rotate(trace, batchSize / (1 << i)));
        }
        return trace;
    }

    virtual const Ciphertext<DCRTPoly> &getZero() const { return m_zeroCache; }
    constexpr size_t getMatrixSize() const { return d; }
};

// For test
template <int d> class TestMatrixOperation : public MatrixOperationBase<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    using MatrixOperationBase<d>::MatrixOperationBase;

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {
        return nullptr;
    }
};

// Secure Outsourced Matrix Computation and Application to Neural Networks,
// CCS 2018
template <int d> class MatrixMult_JKLS18 : public MatrixOperationBase<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;
    using MatrixOperationBase<d>::generateShiftingMsk;
    using MatrixOperationBase<d>::columnShifting;

  public:
    using MatrixOperationBase<d>::MatrixOperationBase;

    std::vector<double> generateSigmaMsk(int k) {
        std::vector<double> u(d * d, 0);
        if (k >= 0) {
            for (int i = d * k; i < d - k + d * k && i < d * d; ++i) {
                u[i] = 1.0;
            }
        } else {
            for (int i = 0; i < d * d; i++) {
                if (i < d + d * (d + k) && i >= -k + d * (d + k))
                    u[i] = 1.0;
                else
                    u[i] = 0.0;
            }
        }
        return u;
    }

    Ciphertext<DCRTPoly> sigmaTransform(const Ciphertext<DCRTPoly> &M) {
        Ciphertext<DCRTPoly> sigma_M = this->getZero()->Clone();

        double squareRootd = sqrt(static_cast<double>(d));
        int squareRootIntd = static_cast<int>(squareRootd);

        int bs;
        if (squareRootIntd * squareRootIntd == 0)
            bs = squareRootIntd;
        else
            bs = round(squareRootd);

        Ciphertext<DCRTPoly> babySteps[bs];
        for (int i = 0; i < bs; i++) {
            babySteps[i] = rot.rotate(M, i);
        }
        for (int i = 1; i < d - bs * (bs - 1); i++) {
            Plaintext pmsk =
                m_cc->MakeCKKSPackedPlaintext(generateSigmaMsk(-d + i));
            m_cc->EvalAddInPlace(sigma_M,
                                 m_cc->EvalMult(rot.rotate(M, i - d), pmsk));
        }
        for (int i = -(bs - 1); i < bs; i++) {
            auto tmp = this->getZero()->Clone();
            for (int j = 0; j < bs; j++) {
                auto msk = generateSigmaMsk(bs * i + j);
                msk = vectorRotate(msk, -bs * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(pmsk, babySteps[j]));
            }
            m_cc->EvalAddInPlace(sigma_M, rot.rotate(tmp, bs * i));
        }

        return sigma_M;
    }

    std::vector<double> generateTauMsk(int k) {
        std::vector<double> msk(d * d, 0);
        for (int i = k; i < d * d; i += d)
            msk[i] = 1;
        return msk;
    }

    Ciphertext<DCRTPoly> tauTransform(const Ciphertext<DCRTPoly> &M) {
        auto tau_M = this->getZero()->Clone();

        double squareRootd = sqrt(static_cast<double>(d));
        int squareRootIntd = static_cast<int>(squareRootd);

        if (squareRootIntd * squareRootIntd == d) {
            Ciphertext<DCRTPoly> babySteps[squareRootIntd];
            for (int i = 0; i < squareRootIntd; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < squareRootIntd; i++) {
                auto tmp = this->getZero()->Clone();

                for (int j = 0; j < squareRootIntd; j++) {
                    auto msk = generateTauMsk(squareRootIntd * i + j);
                    msk = vectorRotate(msk, -squareRootIntd * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk);
                    m_cc->EvalAddInPlace(tmp,
                                         m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M,
                                     rot.rotate(tmp, squareRootIntd * d * i));
            }
        } else {
            int steps = round(squareRootd);

            Ciphertext<DCRTPoly> babySteps[steps];
            for (int i = 0; i < steps; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < d - steps * (steps - 1); i++) {
                Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                    generateTauMsk(steps * (steps - 1) + i));
                m_cc->EvalAddInPlace(
                    tau_M,
                    m_cc->EvalMult(rot.rotate(M, (steps * (steps - 1) + i) * d),
                                   pmsk));
            }
            for (int i = 0; i < steps - 1; i++) {
                auto tmp = this->getZero()->Clone();

                for (int j = 0; j < steps; j++) {
                    auto msk = generateTauMsk(steps * i + j);
                    msk = vectorRotate(msk, -steps * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk);
                    m_cc->EvalAddInPlace(tmp,
                                         m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, steps * d * i));
            }
        }

        return tau_M;
    }

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {
        auto sigma_A = sigmaTransform(matrixA);
        auto tau_B = tauTransform(matrixB);
        auto matrixC = m_cc->EvalMultAndRelinearize(sigma_A, tau_B);

        for (int i = 1; i < d; i++) {
            auto shifted_A = columnShifting(sigma_A, i);
            tau_B = rot.rotate(tau_B, d);
            m_cc->EvalAddInPlace(
                matrixC, m_cc->EvalMultAndRelinearize(shifted_A, tau_B));
        }

        return matrixC;
    }
};

// On Matrix Multiplication with Homomorphic Encryption, CCSW 2022
template <int d> class MatrixMult_RT22 : public MatrixOperationBase<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    using MatrixOperationBase<d>::MatrixOperationBase;

    std::vector<double> generatePhiMsk(int k) {
        std::vector<double> v(d * d * d, 0);
        for (int i = k; i < d * d; i += d) {
            v[i] = 1;
        }
        return v;
    }

    std::vector<double> generatePsiMsk(int k) {
        std::vector<double> v(d * d * d, 0);
        for (int i = k; i < k + d; i++) {
            v[i] = 1;
        }
        return v;
    }

    Ciphertext<DCRTPoly> algoA(const Ciphertext<DCRTPoly> &M) {
        auto A_hat = this->getZero()->Clone();
        for (int i = 0; i < d; i++) {
            auto L_i = m_cc->EvalMult(
                M, m_cc->MakeCKKSPackedPlaintext(generatePhiMsk(i)));
            m_cc->EvalAddInPlace(A_hat, rot.rotate(L_i, i * (1 - d * d)));
        }
        for (int i = 0; i < log2(d); i++) {
            m_cc->EvalAddInPlace(A_hat, rot.rotate(A_hat, -(1 << i)));
        }
        return A_hat;
    }

    Ciphertext<DCRTPoly> algoB(const Ciphertext<DCRTPoly> &M) {
        auto B_hat = this->getZero()->Clone();
        for (int i = 0; i < d; i++) {
            auto R_i = m_cc->EvalMult(
                M, m_cc->MakeCKKSPackedPlaintext(generatePsiMsk(i * d)));
            m_cc->EvalAddInPlace(B_hat, rot.rotate(R_i, i * (d - d * d)));
        }
        for (int i = 0; i < log2(d); i++) {
            m_cc->EvalAddInPlace(B_hat, rot.rotate(B_hat, -d * (1 << i)));
        }
        return B_hat;
    }

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {
        auto A_hat = algoA(matrixA);
        auto B_hat = algoB(matrixB);
        auto matrixC = m_cc->EvalMultAndRelinearize(A_hat, B_hat);

        for (int i = 1; i <= log2(d); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                 rot.rotate(matrixC, (d * d * d) / (1 << i)));
        }
        return matrixC;
    }

    // vectors of length 4
    std::vector<Ciphertext<DCRTPoly>> eval_mult_strassen(
        const std::vector<Ciphertext<DCRTPoly>> &splited_MatrixA,
        const std::vector<Ciphertext<DCRTPoly>> &splited_MatrixB) {
        std::vector<Ciphertext<DCRTPoly>> splited_MatrixC;

        // M1 = (A0 + A3)(B0 + B3)
        auto M1 =
            eval_mult(m_cc->EvalAdd(splited_MatrixA[0], splited_MatrixA[3]),
                      m_cc->EvalAdd(splited_MatrixB[0], splited_MatrixB[3]));
        // M2 = (A2 + A3)B0
        auto M2 =
            eval_mult(m_cc->EvalAdd(splited_MatrixA[2], splited_MatrixA[3]),
                      splited_MatrixB[0]);
        // M3 = A0(B1 - B3)
        auto M3 =
            eval_mult(splited_MatrixA[0],
                      m_cc->EvalSub(splited_MatrixB[1], splited_MatrixB[3]));
        // M4 = A3(B2 - B0)
        auto M4 =
            eval_mult(splited_MatrixA[3],
                      m_cc->EvalSub(splited_MatrixB[2], splited_MatrixB[0]));
        // M5 = (A0 + A1)B3
        auto M5 =
            eval_mult(m_cc->EvalAdd(splited_MatrixA[0], splited_MatrixA[1]),
                      splited_MatrixB[3]);
        // M6 = (A2 - A0)(B0 + B1)
        auto M6 =
            eval_mult(m_cc->EvalSub(splited_MatrixA[2], splited_MatrixA[0]),
                      m_cc->EvalAdd(splited_MatrixB[0], splited_MatrixB[1]));
        // M7 = (A1 - A3)(B2 + B3)
        auto M7 =
            eval_mult(m_cc->EvalSub(splited_MatrixA[1], splited_MatrixA[3]),
                      m_cc->EvalAdd(splited_MatrixB[2], splited_MatrixB[3]));
        // C0 = M1 + M4 - M5 + M7
        splited_MatrixC.push_back(
            m_cc->EvalAdd(m_cc->EvalSub(m_cc->EvalAdd(M1, M4), M5), M7));
        // C1 = M3 + M5
        splited_MatrixC.push_back(m_cc->EvalAdd(M3, M5));
        // C2 = M2 + M4
        splited_MatrixC.push_back(m_cc->EvalAdd(M2, M4));
        // C3 = M1 - M2 + M3 + M6
        splited_MatrixC.push_back(
            m_cc->EvalAdd(m_cc->EvalAdd(m_cc->EvalSub(M1, M2), M3), M6));

        return splited_MatrixC;
    }
};

// Secure and Efficient Outsourced Matrix Multiplication with Homomorphic
// Encryption, Indocrypt 2024
template <int d> class MatrixMult_AS24 : public MatrixOperationBase<d> {
  private:
    int max_batch;
    int B;

  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    int s;

  public:
    MatrixMult_AS24(std::shared_ptr<Encryption> enc, CryptoContext<DCRTPoly> cc,
                    PublicKey<DCRTPoly> publicKey, std::vector<int> rotIndices)
        : MatrixOperationBase<d>(enc, cc, publicKey,
                                 rotIndices) // Parent initialization
    {
        /***  batch size = d*d*s ***/
        max_batch = this->m_cc->GetRingDimension() / 2;
        s = std::min(max_batch / d / d, d);
        B = d / s;
    }

    // k: col number 0~d-1
    std::vector<double> generatePhiMsk(int k) {
        std::vector<double> msk(d * d * s, 0);

        for (int i = k; i < d * d * s; i += d) {
            msk[i] = 1;
        }
        return msk;
    }

    // k: row number 0~d-1
    std::vector<double> generatePsiMsk(int k) {
        std::vector<double> msk(d * d * s, 0);

        for (int i = 0; i < s; i++) {
            for (int j = i * d * d + k * d; j < i * d * d + k * d + d; j++) {
                msk[j] = 1;
            }
        }
        return msk;
    }

    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly> &matA,
                                   const Ciphertext<DCRTPoly> &matB) override {
        int num_slots = d * d * s;
        auto matrixC = this->getZero()->Clone();

        auto matrixA = matA->Clone();
        auto matrixB = matB->Clone();

        Ciphertext<DCRTPoly> Tilde_A[B];
        Ciphertext<DCRTPoly> Tilde_B[B];

        for (int i = 0; i < log2(s); i++) {
            auto tmp = rot.rotate(matrixA, (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixA, tmp);
        }
        for (int i = 0; i < log2(s); i++) {
            auto tmp = rot.rotate(matrixB, d * (1 << i) - d * d * (1 << i));
            m_cc->EvalAddInPlace(matrixB, tmp);
        }

        for (int i = 0; i < B; i++) {
            auto phi_si = m_cc->MakeCKKSPackedPlaintext(
                generatePhiMsk(s * i), 1, 0, nullptr, num_slots);
            auto tmp = m_cc->EvalMult(matrixA, phi_si);
            tmp = rot.rotate(tmp, s * i);
            for (int j = 0; j < log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j)));
            }
            Tilde_A[i] = tmp;
        }

        for (int i = 0; i < B; i++) {
            auto psi_si = m_cc->MakeCKKSPackedPlaintext(
                generatePsiMsk(s * i), 1, 0, nullptr, num_slots);
            auto tmp = m_cc->EvalMult(matrixB, psi_si);
            tmp = rot.rotate(tmp, s * i * d);
            for (int j = 0; j < log2(d); j++) {
                m_cc->EvalAddInPlace(tmp, rot.rotate(tmp, -(1 << j) * d));
            }
            Tilde_B[i] = tmp;
        }

        for (int i = 0; i < B; i++) {
            m_cc->EvalAddInPlace(
                matrixC, m_cc->EvalMultAndRelinearize(Tilde_A[i], Tilde_B[i]));
        }

        for (int i = 0; i < log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                 rot.rotate(matrixC, (d * d) * (1 << i)));
        }
        matrixC->SetSlots(d * d);
        return matrixC;
    }

    Ciphertext<DCRTPoly> clean(const Ciphertext<DCRTPoly> &M){
        std::vector<double> msk(d*d*s, 0.0);          
        for(int i=0; i<d*d; i++){
            msk[i] = 1.0;
        }                      
        auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d*d*s);

        return m_cc->EvalMult(M, pmsk);
    }

    Ciphertext<DCRTPoly> eval_mult_and_clean(const Ciphertext<DCRTPoly> &matA,
                                   const Ciphertext<DCRTPoly> &matB){

        auto matC = eval_mult(matA, matB);
        return clean(matC);
    }
    
};

// Our proposed method, column-based approach
template <int d> class MatrixMult_newCol : public MatrixOperationBase<d> {
  private:
    int max_batch;
    int s;
    int B;
    int ng; // giant-step
    int nb; // baby-step
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixMult_newCol(std::shared_ptr<Encryption> enc,
                      CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> publicKey,
                      std::vector<int> rotIndices)
        : MatrixOperationBase<d>(enc, cc, publicKey,
                                 rotIndices) // Parent initialization
    {
        /***  batch size = d*d*s ***/
        max_batch = this->m_cc->GetRingDimension() / 2;
        s = std::min(max_batch / d / d, d);
        B = d / s;
        if (max_batch == 1 << 13) {
            /*
                This configuration is only for single multiplication,
                where N=1<<14, and max_batch=1<<13
            */
            switch (d) {
            case 4:
                this->ng = 2;
                this->nb = 2;
                break;
            case 8:
                this->ng = 2;
                this->nb = 4;
                break;
            case 16:
                this->ng = 4;
                this->nb = 4;
                break;
            case 32:
                this->ng = 2;
                this->nb = 16;
                break;
            case 64:
                this->ng = 1;
                this->nb = 64;
                break;
            default:
                break;
            }
        } else if (max_batch == 1 << 15) {
            /*
                This configuration is only for 10 multiplication,
                where N=1<<16, and max_batch=1<<15
            */
            switch (d) {
            case 4:
                this->ng = 2;
                this->nb = 2;
                break;
            case 8:
                this->ng = 2;
                this->nb = 4;
                break;
            case 16:
                this->ng = 4;
                this->nb = 4;
                break;
            case 32:
                this->ng = 4;
                this->nb = 8;
                break;
            case 64:
                this->ng = 2;
                this->nb = 32;
                break;
            default:
                break;
            }
        } else if (max_batch == 1 << 16) {
            /*
                This configuration is only for 10 multiplication,
                where N=1<<17, and max_batch=1<<16
            */
            switch (d) {
            case 4:
                this->ng = 2;
                this->nb = 2;
                break;
            case 8:
                this->ng = 2;
                this->nb = 4;
                break;
            case 16:
                this->ng = 4;
                this->nb = 4;
                break;
            case 32:
                this->ng = 4;
                this->nb = 8;
                break;
            case 64:
                this->ng = 4;
                this->nb = 16;
                break;
            default:
                break;
            }
        } else {
            this->ng = -1;
            this->nb = -1;
        }
    }
    std::vector<double> generateMaskVector(int batch_size, int k) {
        std::vector<double> result(batch_size, 0.0);
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

    std::vector<double> genBatchDiagVector(int s, int k, int diag_index) {
        std::vector<double> result;
        result.reserve(d * d * s);

        for (int i = 0; i < s; ++i) {
            std::vector<double> diag_vector = genDiagVector(k + i, diag_index);
            result.insert(result.end(), diag_vector.begin(), diag_vector.end());
        }

        return result;
    }

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {
        int num_slots = s * d * d;
        auto matrixC = this->getZero()->Clone();

        Ciphertext<DCRTPoly> babyStepsOfA[nb];

        // nb rotations required
        for (int i = 0; i < nb; i++) {
            babyStepsOfA[i] = rot.rotate(matrixA, i);
        }

        for (int i = 0; i < B; i++) {
            auto batched_rotations_B = this->getZero()->Clone();
            for (int j = 0; j < s; j++) {
                auto rotated_of_B = rot.rotate(matrixB, i * s * d + j * d);
                rotated_of_B->SetSlots(num_slots);
                m_cc->EvalAddInPlace(
                    batched_rotations_B,
                    m_cc->EvalMult(rotated_of_B,
                                   m_cc->MakeCKKSPackedPlaintext(
                                       generateMaskVector(num_slots, j), 1, 0,
                                       nullptr, num_slots)));
            }

            auto diagA = this->getZero()->Clone();
            for (int k = -ng; k < ng; k++) {
                if (k < 0) {
                    auto tmp = this->getZero()->Clone();
                    auto babyStep = (k == -ng) ? 1 : 0;
                    for (int j = d * d + k * nb + 1 + babyStep;
                         j <= d * d + (k + 1) * nb; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                        m_cc->EvalAddInPlace(
                            tmp, m_cc->EvalMult(babyStepsOfA[babyStep],
                                                m_cc->MakeCKKSPackedPlaintext(
                                                    rotated_plain_vec, 1, 0,
                                                    nullptr, num_slots)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb));
                } else { // k>=0
                    auto tmp = this->getZero()->Clone();
                    auto babyStep = 0;
                    for (int j = k * nb + 1; j <= (k + 1) * nb; j++) {
                        auto rotated_plain_vec = vectorRotate(
                            genBatchDiagVector(s, i * s, j), -k * nb);
                        m_cc->EvalAddInPlace(
                            tmp, m_cc->EvalMult(babyStepsOfA[babyStep],
                                                m_cc->MakeCKKSPackedPlaintext(
                                                    rotated_plain_vec, 1, 0,
                                                    nullptr, num_slots)));
                        babyStep++;
                    }
                    m_cc->EvalAddInPlace(diagA, rot.rotate(tmp, k * nb));
                }
            }
            m_cc->EvalAddInPlace(matrixC,
                                 m_cc->EvalMult(diagA, batched_rotations_B));
        }
        for (int i = 1; i <= log2(s); i++) {
            m_cc->EvalAddInPlace(matrixC,
                                 rot.rotate(matrixC, num_slots / (1 << i)));
        }
        matrixC->SetSlots(d * d);

        return matrixC;
    }
};

// Our proposed method, row-based approach
template <int d> class MatrixMult_newRow : public MatrixOperationBase<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;
    using MatrixOperationBase<d>::generateShiftingMsk;
    using MatrixOperationBase<d>::columnShifting;

  public:
    using MatrixOperationBase<d>::MatrixOperationBase;

    Ciphertext<DCRTPoly> eval_mult(const Ciphertext<DCRTPoly> &matrixA,
                                   const Ciphertext<DCRTPoly> &matrixB) {
        int l = 2;

        Ciphertext<DCRTPoly> rots_B[l];
        // require l-1 rotation
        for (int i = 0; i < l; i++) {
            rots_B[i] = rot.rotate(matrixB, d * i);
        }

        // Step 1 : concatenationg diagonals
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }
        auto msk1 = vectorRotate(msk, d);

        // Step 2 : matrix multiplication
        auto diag_B =
            m_cc->EvalMult(m_cc->MakeCKKSPackedPlaintext(msk), matrixB);
        for (int i = 1; i < l; i++) {
            m_cc->EvalAddInPlace(
                diag_B,
                m_cc->EvalMult(m_cc->MakeCKKSPackedPlaintext(msk1), rots_B[i]));
            msk1 = vectorRotate(msk1, d);
        }
        for (int j = log2(l); j < log2(d); j++) {
            auto rot_diag_B = rot.rotate(diag_B, -d * (1 << j));
            m_cc->EvalAddInPlace(diag_B, rot_diag_B);
        }

        auto matrixC = m_cc->EvalMultAndRelinearize(diag_B, matrixA);
        for (int i = 1; i < d; i++) {
            msk1 = msk;
            msk = vectorRotate(msk, -d);
            diag_B =
                m_cc->EvalMult(m_cc->MakeCKKSPackedPlaintext(msk), matrixB);

            for (int j = 1; j < l; j++) {
                diag_B = m_cc->EvalAdd(
                    diag_B, m_cc->EvalMult(m_cc->MakeCKKSPackedPlaintext(msk1),
                                           rots_B[j]));
                msk1 = vectorRotate(msk1, d);
            }
            for (int j = log2(l); j < log2(d); j++) {
                auto rot_diag_B = rot.rotate(diag_B, -d * (1 << j));
                m_cc->EvalAddInPlace(diag_B, rot_diag_B);
            }
            m_cc->EvalAddInPlace(matrixC,
                                 m_cc->EvalMultAndRelinearize(
                                     diag_B, columnShifting(matrixA, i)));
        }

        return matrixC;
    }
};