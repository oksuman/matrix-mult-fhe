#pragma once

#include "matrix_algo_singlePack.h"
#include <memory>
#include <openfhe.h>

using namespace lbcrypto;

// Scalar inversion iterations (unified)
constexpr int SCALAR_INV_ITERATIONS = 1;

// Base class for matrix inverse operations
template <int d> class MatrixInverseBase {
  protected:
    int r;
    int depth;
    int scalar_inv_iter;

    virtual std::vector<double> initializeIdentityMatrix(size_t dim) {
        std::vector<double> identity(dim * dim, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

  public:
    MatrixInverseBase(int iterations, int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : r(iterations), depth(multDepth), scalar_inv_iter(scalarInvIter) {}
    virtual ~MatrixInverseBase() = default;

    virtual Ciphertext<DCRTPoly>
    eval_inverse(const Ciphertext<DCRTPoly> &M) = 0;

    // trace 계산 및 eval_scalar_inverse 없이
    // upperbound (1/d^2) 로 직접 scaling 하는 간소화된 inversion.
    // eval_transpose_scaled 를 이용해 transpose + scaling 을 한 번에 처리한다.
    virtual Ciphertext<DCRTPoly>
    eval_inverse_simple(const Ciphertext<DCRTPoly> &M) = 0;
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
                         int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : MatrixInverseBase<d>(iterations, multDepth, scalarInvIter), MatrixMult_JKLS18<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);

        // Bootstrap before scalar inverse if needed
        if ((int)trace->GetLevel() >= this->depth - 10) {
            trace = m_cc->EvalBootstrap(trace, 2, 18);
        }
        auto trace_reciprocal = this->eval_scalar_inverse(trace, d * d, this->scalar_inv_iter, d * d);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 3) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 3) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }

    // trace/eval_scalar_inverse 없이 1/d^2 upperbound 로 직접 scaling.
    // Y0 = eval_transpose_scaled(M, 1/d^2)
    // A0 = I - eval_mult(M, Y0)  (= I - (1/d^2)*M*M^T)
    Ciphertext<DCRTPoly> eval_inverse_simple(const Ciphertext<DCRTPoly> &M) override {
        double scaleFactor = 1.0 / static_cast<double>(d * d);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto Y = this->eval_transpose_scaled(M, scaleFactor);
        auto A_bar = this->m_cc->EvalSub(pI, this->eval_mult(M, Y));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 3 ||
                (int)A_bar->GetLevel() >= this->depth - 3) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 3 ||
            (int)A_bar->GetLevel() >= this->depth - 3) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }
};

// RT22
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
                       int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : MatrixInverseBase<d>(iterations, multDepth, scalarInvIter), MatrixMult_RT22<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);
        MM_transposed->SetSlots(d * d);

        auto trace = this->eval_trace(MM_transposed, d * d);

        // Bootstrap before scalar inverse if needed
        if ((int)trace->GetLevel() >= this->depth - 10) {
            trace = m_cc->EvalBootstrap(trace, 2, 18);
        }
        auto trace_reciprocal = this->eval_scalar_inverse(trace, d * d, this->scalar_inv_iter, d * d);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2) {
            A_bar->SetSlots(d * d);
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y->SetSlots(d * d);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }

    // trace/eval_scalar_inverse 없이 1/d^2 upperbound 로 직접 scaling.
    Ciphertext<DCRTPoly> eval_inverse_simple(const Ciphertext<DCRTPoly> &M) override {
        double scaleFactor = 1.0 / static_cast<double>(d * d);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto Y = this->eval_transpose_scaled(M, scaleFactor);
        auto MM_scaled = this->eval_mult(M, Y);
        MM_scaled->SetSlots(d * d);
        auto A_bar = this->m_cc->EvalSub(pI, MM_scaled);

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2 ||
                (int)A_bar->GetLevel() >= this->depth - 2) {
                A_bar->SetSlots(d * d);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y->SetSlots(d * d);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2 ||
            (int)A_bar->GetLevel() >= this->depth - 2) {
            A_bar->SetSlots(d * d);
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y->SetSlots(d * d);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// AR24
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
                       int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : MatrixInverseBase<d>(iterations, multDepth, scalarInvIter), MatrixMult_AR24<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);

        auto M_transposed = this->eval_transpose(M);

        auto M_expanded = M->Clone();
        M_expanded->SetSlots(d * d * this->s);
        M_expanded = this->clean(M_expanded);
        auto Mt_expanded = M_transposed->Clone();
        Mt_expanded->SetSlots(d * d * this->s);
        Mt_expanded = this->clean(Mt_expanded);

        auto MM_transposed = this->eval_mult(M_expanded, Mt_expanded);

        auto trace = this->eval_trace(MM_transposed, d * d);

        if ((int)trace->GetLevel() >= this->depth - 10) {
            trace->SetSlots(d * d);
            trace = m_cc->EvalBootstrap(trace, 2, 18);
        }
        auto trace_reciprocal = this->eval_scalar_inverse(trace, d * d, this->scalar_inv_iter, d * d);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 3) {
                Y = m_cc->EvalBootstrap(Y, 2, 18);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            }
            auto pI_level = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d * d);
            auto I_plus_A = this->m_cc->EvalAdd(pI_level, A_bar);

            I_plus_A->SetSlots(d * d * this->s);
            I_plus_A = this->clean(I_plus_A);
            Y->SetSlots(d * d * this->s);
            Y = this->clean(Y);
            Y = this->eval_mult(Y, I_plus_A);

            A_bar->SetSlots(d * d * this->s);
            A_bar = this->clean(A_bar);
            auto A_bar_copy = A_bar->Clone();
            A_bar = this->eval_mult(A_bar, A_bar_copy);
        }
        if ((int)Y->GetLevel() >= this->depth - 3) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
        }
        auto pI_final = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d * d);
        auto I_plus_A_final = this->m_cc->EvalAdd(pI_final, A_bar);

        I_plus_A_final->SetSlots(d * d * this->s);
        I_plus_A_final = this->clean(I_plus_A_final);
        Y->SetSlots(d * d * this->s);
        Y = this->clean(Y);
        Y = this->eval_mult(Y, I_plus_A_final);
        return Y;
    }

    // trace/eval_scalar_inverse 없이 1/d^2 upperbound 로 직접 scaling.
    // eval_transpose_scaled 로 Y0 = (1/d^2)*M^T 를 한 번에 계산하고,
    // eval_mult(M, Y0) 으로 (1/d^2)*M*M^T 를 구한다.
    Ciphertext<DCRTPoly> eval_inverse_simple(const Ciphertext<DCRTPoly> &M) override {
        double scaleFactor = 1.0 / static_cast<double>(d * d);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);

        // Y0 = (1/d^2)*M^T (scaled transpose)
        auto Y = this->eval_transpose_scaled(M, scaleFactor);

        // eval_mult(M, Y0) = (1/d^2)*M*M^T — AR24는 clean() 필요
        auto M_expanded = M->Clone();
        M_expanded->SetSlots(d * d * this->s);
        M_expanded = this->clean(M_expanded);
        auto Y_expanded = Y->Clone();
        Y_expanded->SetSlots(d * d * this->s);
        Y_expanded = this->clean(Y_expanded);

        auto MM_scaled = this->eval_mult(M_expanded, Y_expanded);
        auto A_bar = this->m_cc->EvalSub(pI, MM_scaled);

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 3) {
                Y = m_cc->EvalBootstrap(Y, 2, 18);
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            }
            auto pI_level = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d * d);
            auto I_plus_A = this->m_cc->EvalAdd(pI_level, A_bar);

            I_plus_A->SetSlots(d * d * this->s);
            I_plus_A = this->clean(I_plus_A);
            Y->SetSlots(d * d * this->s);
            Y = this->clean(Y);
            Y = this->eval_mult(Y, I_plus_A);

            A_bar->SetSlots(d * d * this->s);
            A_bar = this->clean(A_bar);
            auto A_bar_copy = A_bar->Clone();
            A_bar = this->eval_mult(A_bar, A_bar_copy);
        }
        if ((int)Y->GetLevel() >= this->depth - 3) {
            Y = m_cc->EvalBootstrap(Y, 2, 18);
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
        }
        auto pI_final = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d * d);
        auto I_plus_A_final = this->m_cc->EvalAdd(pI_final, A_bar);

        I_plus_A_final->SetSlots(d * d * this->s);
        I_plus_A_final = this->clean(I_plus_A_final);
        Y->SetSlots(d * d * this->s);
        Y = this->clean(Y);
        Y = this->eval_mult(Y, I_plus_A_final);
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
                         int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : MatrixInverseBase<d>(iterations, multDepth, scalarInvIter), MatrixMult_newCol<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);

        if ((int)trace->GetLevel() >= this->depth - 10) {
            trace = m_cc->EvalBootstrap(trace, 2, 18);
        }
        auto trace_reciprocal = this->eval_scalar_inverse(trace, d * d, this->scalar_inv_iter, d * d);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }

    // trace/eval_scalar_inverse 없이 1/d^2 upperbound 로 직접 scaling.
    Ciphertext<DCRTPoly> eval_inverse_simple(const Ciphertext<DCRTPoly> &M) override {
        double scaleFactor = 1.0 / static_cast<double>(d * d);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, d * d);

        auto Y = this->eval_transpose_scaled(M, scaleFactor);
        auto A_bar = this->m_cc->EvalSub(pI, this->eval_mult(M, Y));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// newRow
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
                         int multDepth, int scalarInvIter = SCALAR_INV_ITERATIONS)
        : MatrixInverseBase<d>(iterations, multDepth, scalarInvIter), MatrixMult_newRow<d>(
                                                           enc, cc, publicKey,
                                                           rotIndices) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        auto trace = this->eval_trace(MM_transposed, d * d);

        // Bootstrap before scalar inverse if needed
        if ((int)trace->GetLevel() >= this->depth - 10) {
            trace = m_cc->EvalBootstrap(trace, 2, 18);
        }
        auto trace_reciprocal = this->eval_scalar_inverse(trace, d * d, this->scalar_inv_iter, d * d);

        auto Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));

        return Y;
    }

    // newRow 는 실험 대상 외이지만 인터페이스 완성을 위해 구현
    Ciphertext<DCRTPoly> eval_inverse_simple(const Ciphertext<DCRTPoly> &M) override {
        double scaleFactor = 1.0 / static_cast<double>(d * d);

        std::vector<double> vI = this->initializeIdentityMatrix(d);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

        auto Y = this->eval_transpose_scaled(M, scaleFactor);
        auto A_bar = this->m_cc->EvalSub(pI, this->eval_mult(M, Y));

        for (int i = 0; i < this->r - 1; i++) {
            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
                Y = m_cc->EvalBootstrap(Y, 2, 18);
            }
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);
        }
        if ((int)Y->GetLevel() >= this->depth - 2) {
            A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);
            Y = m_cc->EvalBootstrap(Y, 2, 18);
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};
