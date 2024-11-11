#pragma once

#include "rotation.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <openfhe.h>
#include <random>
#include <vector>

using namespace lbcrypto;

template <int d> class MatrixMultiPackBase {
  protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;
    std::unique_ptr<RotationComposer> rot;
    Ciphertext<DCRTPoly> m_zeroCache;

    virtual Ciphertext<DCRTPoly> createZeroCache() {
        std::vector<double> zeroVec(d, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

  public:
    MatrixMultiPackBase(std::shared_ptr<Encryption> enc,
                        CryptoContext<DCRTPoly> cc,
                        PublicKey<DCRTPoly> publicKey,
                        const std::vector<int> &rotIndices)
        : m_enc(enc), m_cc(cc), m_publicKey(publicKey) {
        rot = std::make_unique<RotationComposer>(cc, rotIndices, d);
        m_zeroCache = createZeroCache();
    }

    virtual ~MatrixMultiPackBase() = default;
    virtual std::vector<Ciphertext<DCRTPoly>>
    eval_mult(const std::vector<Ciphertext<DCRTPoly>> &matrixA,
              const std::vector<Ciphertext<DCRTPoly>> &matrixB) = 0;
    constexpr size_t getMatrixSize() const { return d; }
};

template <int d> class MatrixInv_diag : public MatrixMultiPackBase<d> {
  protected:
    using MatrixMultiPackBase<d>::rot;
    using MatrixMultiPackBase<d>::m_cc;
    int r; // iteration count for inverse calculation

  public:
    using MatrixMultiPackBase<d>::MatrixMultiPackBase;

    MatrixInv_diag(std::shared_ptr<Encryption> enc, CryptoContext<DCRTPoly> cc,
                   PublicKey<DCRTPoly> publicKey,
                   const std::vector<int> &rotIndices, int iterCount)
        : MatrixMultiPackBase<d>(enc, cc, publicKey, rotIndices), r(iterCount) {
    }

    std::vector<Ciphertext<DCRTPoly>>
    eval_add_plain(const std::vector<Ciphertext<DCRTPoly>> &matrixA,
                   const std::vector<Plaintext> &matrixB) {
        std::vector<Ciphertext<DCRTPoly>> matrixC;

        for (int i = 0; i < d; i++) {
            auto diag = this->m_cc->EvalAdd(matrixA[i], matrixB[i]);
            matrixC.push_back(diag);
        }

        return matrixC;
    }

    std::vector<Ciphertext<DCRTPoly>>
    eval_mult(const std::vector<Ciphertext<DCRTPoly>> &matrixA,
              const std::vector<Ciphertext<DCRTPoly>> &matrixB) override {

        std::vector<Ciphertext<DCRTPoly>> matrixC;
        for (int i = 0; i < d; i++) {
            auto diag = this->m_zeroCache->Clone();
            for (int j = 0; j < d; j++) {
                this->m_cc->EvalAddInPlace(
                    diag, this->m_cc->EvalMultAndRelinearize(
                              matrixA[j],
                              this->rot->rotate(matrixB[(i - j + d) % d], j)));
            }
            matrixC.push_back(diag);
        }
        return matrixC;
    }

    std::vector<Ciphertext<DCRTPoly>>
    eval_transpose(const std::vector<Ciphertext<DCRTPoly>> &M) {
        std::vector<Ciphertext<DCRTPoly>> result;

        for (int i = 0; i < d; i++) {
            result.push_back(this->rot->rotate(M[(d - i) % d], i));
        }
        return result;
    }

    std::vector<Plaintext> initializeIdentityMatrix() {
        std::vector<Plaintext> matrix;
        std::vector<double> diag(d, 1.0);

        Plaintext p = this->m_cc->MakeCKKSPackedPlaintext(diag);
        matrix.push_back(p);

        for (int i = 1; i < d; ++i) {
            std::vector<double> diag(d, 0.0);
            Plaintext p = this->m_cc->MakeCKKSPackedPlaintext(diag);
            matrix.push_back(p);
        }
        return matrix;
    }

    std::vector<Ciphertext<DCRTPoly>>
    eval_inverse(const std::vector<Ciphertext<DCRTPoly>> &M) {
        auto M_transposed = eval_transpose(M);
        auto MM_transposed = eval_mult(M, M_transposed);

        auto trace = MM_transposed[0]->Clone();
        for (int i = 1; i <= log2(d); i++) {
            m_cc->EvalAddInPlace(trace, rot->rotate(trace, d / (1 << i)));
        }

        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);

        std::vector<Ciphertext<DCRTPoly>> Y;
        for (int i = 0; i < d; i++) {
            Y.push_back(this->m_cc->EvalMultAndRelinearize(M_transposed[i],
                                                           trace_reciprocal));
        }

        std::vector<Ciphertext<DCRTPoly>> A;
        auto I = initializeIdentityMatrix();
        for (int i = 0; i < d; i++) {
            A.push_back(this->m_cc->EvalSub(
                I[i], this->m_cc->EvalMultAndRelinearize(MM_transposed[i],
                                                         trace_reciprocal)));
        }

        for (int i = 0; i < r; i++) {
            Y = eval_mult(Y, eval_add_plain(A, I));
            A = eval_mult(A, A);
        }
        return Y;
    }
};