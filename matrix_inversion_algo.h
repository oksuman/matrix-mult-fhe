#pragma once

#include "matrix_algo_singlePack.h"
#include <memory>
#include <openfhe.h>

using namespace lbcrypto;

// Base class for matrix inverse operations
template <int d> class MatrixInverseBase {
  protected:
    static constexpr int r = 23;
    static constexpr int depth = 34;

    virtual std::vector<double> initializeIdentityMatrix() {
        std::vector<double> identity(d*d, 0.0);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < d; i++) {
            identity[i * d + i] = 1.0;
        }
        return identity;
    }

  public:
    MatrixInverseBase() {} 
    virtual ~MatrixInverseBase() = default;

    virtual Ciphertext<DCRTPoly>
    eval_inverse(const Ciphertext<DCRTPoly> &M) = 0;
};

// Implementation using newCol matrix multiplication
template <int d>
class MatrixInverse_newColOpt : public MatrixInverseBase<d>,
                             public MatrixMult_newColOpt<d> {
  protected:
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixInverse_newColOpt(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey)
        : MatrixInverseBase<d>(), MatrixMult_newColOpt<d>(enc, cc, publicKey) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix();
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);
        Plaintext debug;

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, 1, (d * d) / 3 + d, 5);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
       
        for (int i = 0; i < this->r - 1; i++) {
            auto identity = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d*d);
            auto A_plus_I = this->m_cc->EvalAdd(identity, A_bar);
            Y = this->eval_mult(Y, A_plus_I);
            A_bar = this->eval_mult(A_bar, A_bar);   

            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

