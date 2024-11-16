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
        std::cout << "transpose" << std::endl;
        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        std::cout << "trace" << std::endl;
        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);

        std::cout << "mult start" << std::endl;
        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r-1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            std::cout << "i: " << i << std::endl;
            A_bar = this->eval_mult(A_bar, A_bar);
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
        // MM_transposed->SetSlots(d*d);

        auto trace = this->eval_trace(MM_transposed, d * d * d);
        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r-1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }

    std::vector<Ciphertext<DCRTPoly>>
    eval_inverse_strassen(const std::vector<Ciphertext<DCRTPoly>> &M) {
        std::vector<double> vI = this->initializeIdentityMatrix(d / 2);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        std::vector<Ciphertext<DCRTPoly>> M_transposed(4);
        for (int i = 0; i < 4; i++) {
            M_transposed[i] = this->eval_transpose(M[i]);
        }

        auto MM_transposed = this->eval_mult_strassen(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed[0], d * d * d);
        for (int i = 1; i < 4; i++) {
            m_cc->EvalAddInPlace(trace,
                                 this->eval_trace(MM_transposed[i], d * d * d));
        }

        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d / 2, d * d, 50);

        std::vector<Ciphertext<DCRTPoly>> Y(4), A_bar(4);
        for (int i = 0; i < 4; i++) {
            Y[i] = this->m_cc->EvalMultAndRelinearize(M_transposed[i],
                                                      trace_reciprocal);
            A_bar[i] = this->m_cc->EvalSub(
                pI, this->m_cc->EvalMultAndRelinearize(MM_transposed[i],
                                                       trace_reciprocal));
        }

        for (int i = 0; i < this->r-1; i++) {
            auto pI_plus_A = A_bar;
            for (int j = 0; j < 4; j++) {
                m_cc->EvalAddInPlace(pI_plus_A[j], pI);
            }
            Y = this->eval_mult_strassen(Y, pI_plus_A);
            A_bar = this->eval_mult_strassen(A_bar, A_bar);
        }
        auto pI_plus_A = A_bar;
        for (int j = 0; j < 4; j++) {
            m_cc->EvalAddInPlace(pI_plus_A[j], pI);
        }
        Y = this->eval_mult_strassen(Y, pI_plus_A);

        return Y;
    }
};

template <int d>
class MatrixInverse_AS24 : public MatrixInverseBase<d>,
                             public MatrixMult_AS24<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    MatrixInverse_AS24(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey,
                         std::vector<int> rotIndices, int iterations,
                         int multDepth)
        : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_AS24<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        std::cout << "transpose" << std::endl;
        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        std::cout << "trace" << std::endl;
        auto trace = this->eval_trace(MM_transposed, d * d * d);
        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);

        std::cout << "trace" << std::endl;
        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
        A_bar->SetSlots(d*d*this->s);
        A_bar = this->clean(A_bar);
        std::cout <<"loop start" << std::endl;

        for (int i = 0; i < this->r-1; i++) {
            if (d >= 16 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
            }
            Y = this->eval_mult_and_clean(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult_and_clean(A_bar, A_bar);
            std::cout << "i: " << i << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }

    Ciphertext<DCRTPoly> eval_inverse_debug(const Ciphertext<DCRTPoly> &M, PrivateKey<DCRTPoly> m_privateKey){
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        Plaintext debug_ptx;
        std::vector<double> debug_vec;

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);


        this->m_cc->Decrypt(m_privateKey, M_transposed, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "M^T: " << std::endl;
        std::cout << debug_vec << std::endl;

        this->m_cc->Decrypt(m_privateKey, MM_transposed, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "MM^T: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto trace = this->eval_trace(MM_transposed, d * d * d);

        this->m_cc->Decrypt(m_privateKey, trace, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "trace: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);

        this->m_cc->Decrypt(m_privateKey, trace_reciprocal, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "trace_reciprocal: " << std::endl;
        std::cout << debug_vec << std::endl;

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
        A_bar->SetSlots(d*d*this->s);
        A_bar = this->clean(A_bar);

        this->m_cc->Decrypt(m_privateKey, Y, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "Y: " << std::endl;
        std::cout << debug_vec << std::endl;

        this->m_cc->Decrypt(m_privateKey, A_bar, &debug_ptx);
        debug_vec = debug_ptx->GetRealPackedValue();
        std::cout << "A_bar: " << std::endl;
        std::cout << debug_vec << std::endl;

        for (int i = 0; i < this->r-1; i++) {
            if (d >= 16 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
            }
            Y = this->eval_mult_and_clean(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult_and_clean(A_bar, A_bar);
             std::cout << i << "-th iteration: " << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl; 
            std::cout << "level of A_bar: " << A_bar->GetLevel() << std::endl; 

            this->m_cc->Decrypt(m_privateKey, Y, &debug_ptx);
            debug_vec = debug_ptx->GetRealPackedValue();
            std::cout << "Y: " << std::endl;
            std::cout << debug_vec << std::endl;

            this->m_cc->Decrypt(m_privateKey, A_bar, &debug_ptx);
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

        std::cout << "mmt level: " << MM_transposed->GetLevel() << std::endl;

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal = this->m_cc->EvalDivide(trace, (d*d)/3 - d, (d*d)/3 + d, 50);

        std::cout << "1/trace level: " << trace_reciprocal->GetLevel() << std::endl;

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));


        for (int i = 0; i < this->r-1; i++) {
            std::cout << "i: " << i << std::endl;
            std::cout << "Y level: " << Y->GetLevel() << std::endl;
            std::cout << "A level: " << A_bar->GetLevel() << std::endl;
            
            if (d >= 16 && (int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
                Y = m_cc->EvalBootstrap(Y, 2, 17);
                std::cout << "bootstrapping required" << std::endl;
                std::cout << "after bootstrapping Y level: " << Y->GetLevel() << std::endl;
                std::cout << "after bootstrapping A level: " << A_bar->GetLevel() << std::endl;
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        std::cout << "final Y level: " << Y->GetLevel() << std::endl;
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
        std::cout << "start" << std::endl;
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);
        std::cout << "transposition" << std::endl;

        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal = this->m_cc->EvalDivide(trace, d, d * d, 50);
        std::cout << "division" << std::endl;

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r-1; i++) {
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
            std::cout << "i: " << i << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }
};
