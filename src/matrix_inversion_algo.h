#pragma once

#include "matrix_algo_singlePack.h"
#include <memory>
#include <openfhe.h>

using namespace lbcrypto;

// Base class for matrix inverse operations
template <int d> class MatrixInverseBase {
  protected:
    int r;
    int depth;
    virtual std::vector<double> initializeIdentityMatrix(size_t dim) {
        std::vector<double> identity(dim * dim, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

  public:
    MatrixInverseBase(int iterations, int multDepth)
        : r(iterations), depth(multDepth) {}
    virtual ~MatrixInverseBase() = default;

    virtual Ciphertext<DCRTPoly>
    eval_inverse(const Ciphertext<DCRTPoly> &M) = 0;
};

// Implementation using JKLS18 matrix multiplication
template <int d>
class MatrixInverse_JKLS18 : public MatrixInverseBase<d>,
                             public MatrixMult_JKLS18<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;
    using MatrixOperationBase<d>::generateShiftingMsk;
    using MatrixOperationBase<d>::columnShifting;

  public:
    MatrixInverse_JKLS18(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey,
                         std::vector<int> rotIndices, int iterations,
                         int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_JKLS18<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
            if ((int)Y->GetLevel() >= this->depth - 3) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }
};

template <int d>
class MatrixInverse_RT22 : public MatrixInverseBase<d>,
                           public MatrixMult_RT22<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    MatrixInverse_RT22(std::shared_ptr<Encryption> enc,
                       CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey,
                       std::vector<int> rotIndices, int iterations,
                       int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_RT22<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if (d >= 8 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

template <int d>
class MatrixInverse_AR24 : public MatrixInverseBase<d>,
                           public MatrixMult_AR24<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    MatrixInverse_AR24(std::shared_ptr<Encryption> enc,
                       CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey,
                       std::vector<int> rotIndices, int iterations,
                       int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_AR24<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d * this->s);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
        A_bar->SetSlots(d * d * this->s);
        A_bar = this->clean(A_bar);

        for (int i = 0; i < this->r - 1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);

            if ((int)Y->GetLevel() >= this->depth - 3) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
                A_bar->SetSlots(d * d * this->s);
                A_bar = this->clean(A_bar);
                Y->SetSlots(d * d * this->s);
                Y = this->clean(Y);
            } else {
                A_bar = this->clean(A_bar);
                Y = this->clean(Y);
            }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }

    Ciphertext<DCRTPoly> eval_inverse_debug(const Ciphertext<DCRTPoly> &M,
                                            PrivateKey<DCRTPoly> m_privateKey) {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        Plaintext debug_ptx;
        std::vector<double> debug_vec;

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        this->m_cc->Decrypt(m_privateKey, M_transposed, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "M^T: " << std::endl;
        std::cout << debug_vec << std::endl;

        this->m_cc->Decrypt(m_privateKey, MM_transposed, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "MM^T: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto trace = this->eval_trace(MM_transposed, d * d * this->s);

        this->m_cc->Decrypt(m_privateKey, trace, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "trace: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto trace_reciprocal = this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 5);

        this->m_cc->Decrypt(m_privateKey, trace_reciprocal, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "trace_reciprocal: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
        A_bar->SetSlots(d * d * this->s);
        A_bar = this->clean(A_bar);

        this->m_cc->Decrypt(m_privateKey, Y, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "Y: " << std::endl;
        std::cout << debug_vec << std::endl;

        this->m_cc->Decrypt(m_privateKey, A_bar, &debug_ptx);
        debug_ptx->SetLength(10);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "A_bar: " << std::endl;
        std::cout << debug_vec << std::endl;

        for (int i = 0; i < this->r - 1; i++) {

            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult_and_clean(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult_and_clean(A_bar, A_bar);
            std::cout << i << "-th iteration: " << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl;
            std::cout << "level of A_bar: " << A_bar->GetLevel() << std::endl;

            this->m_cc->Decrypt(m_privateKey, Y, &debug_ptx);
            debug_vec = debug_ptx->GetRealPackedValue();
            debug_ptx->SetLength(10);
            std::cout << "Y: " << std::endl;
            std::cout << debug_vec << std::endl;

            this->m_cc->Decrypt(m_privateKey, A_bar, &debug_ptx);
            debug_ptx->SetLength(10);
            debug_vec = debug_ptx->GetRealPackedValue();
            std::cout << "A_bar: " << std::endl;
            std::cout << debug_vec << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// Implementation using newCol matrix multiplication
template <int d>
class MatrixInverse_newCol : public MatrixInverseBase<d>,
                             public MatrixMult_newCol<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixInverse_newCol(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey,
                         std::vector<int> rotIndices, int iterations,
                         int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_newCol<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if (d >= 8 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// row-based approach
template <int d>
class MatrixInverse_newRow : public MatrixInverseBase<d>,
                             public MatrixMult_newRow<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixInverse_newRow(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey,
                         std::vector<int> rotIndices, int iterations,
                         int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_newRow<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if (d >= 8 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }
};
