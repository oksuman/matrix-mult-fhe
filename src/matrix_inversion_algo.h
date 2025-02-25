#pragma once

#include "matrix_algo_singlePack.h"
#include <memory>
#include <openfhe.h>

using namespace lbcrypto;

// Base class for matrix inverse operations
template <int d> class MatrixInverseBase {
  protected:
    static constexpr int r = 22;
    static constexpr int depth = 49; // 2r+9 48

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

template <int d>
class MatrixInverse_AS24Opt : public MatrixInverseBase<d>,
                           public MatrixMult_AS24Opt<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;

  public:
    MatrixInverse_AS24Opt(std::shared_ptr<Encryption> enc,
                       CryptoContext<DCRTPoly> cc,
                       PublicKey<DCRTPoly> publicKey)
        : MatrixInverseBase<d>(), MatrixMult_AS24Opt<d>(enc, cc, publicKey) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        std::vector<double> vI = this->initializeIdentityMatrix();
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, this->max_batch);

        auto M_transposed = this->eval_transpose(M);

        auto m_M = M->Clone();
        m_M->SetSlots(this->max_batch);
        m_M = this->clean(m_M);
        M_transposed->SetSlots(this->max_batch);
        M_transposed = this->clean(M_transposed);

        auto MM_transposed = this->eval_mult(m_M, M_transposed);
        
        auto trace = this->eval_trace(MM_transposed, d * d);
        // auto trace_reciprocal =
        //     this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 5);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, 1, (d * d) / 3 + d, 5);

        trace_reciprocal->SetSlots(this->max_batch);
        MM_transposed->SetSlots(this->max_batch);
        MM_transposed = this->clean(MM_transposed);

        Ciphertext<DCRTPoly> A_bar;
        Ciphertext<DCRTPoly> Y;

        Y = this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        A_bar = this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(MM_transposed, trace_reciprocal));
        
        std::cout << "level of Y: " << Y->GetLevel() << std::endl;
        std::cout << "level of A: " << A_bar->GetLevel() << std::endl;

        for (int i = 0; i < this->r - 1; i++) {
            auto identity = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, this->max_batch);
            auto tmp = this->m_cc->EvalAdd(pI, A_bar);
            Y = this->eval_mult(Y, tmp);
            A_bar = this->eval_mult(A_bar, A_bar);   

            if ((int)Y->GetLevel() >= this->depth - 3) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);   
                Y = m_cc->EvalBootstrap(Y, 2, 18);     
            } 
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    A_bar->SetSlots(this->max_batch);
                    A_bar = this->clean(A_bar);
                }
                #pragma omp section
                {
                    Y->SetSlots(this->max_batch);
                    Y = this->clean(Y);
                }
            }
            std::cout << "current interation: " << i << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl;
            std::cout << "level of A: " << A_bar->GetLevel() << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }

    Ciphertext<DCRTPoly> eval_inverse_debug(const Ciphertext<DCRTPoly> &M, PrivateKey<DCRTPoly> sk) {
        Plaintext debug;

        std::vector<double> vI = this->initializeIdentityMatrix();
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, this->max_batch);

        auto m_M = M->Clone();
        m_M->SetSlots(this->max_batch);
        m_M = this->clean(m_M);
        auto M_transposed = this->eval_transpose(m_M);

        std::cout << "slots of M: " << m_M->GetSlots() << std::endl;
        m_cc->Decrypt(m_M, sk, &debug);
        debug->SetLength(10);
        std::cout << "M: " << debug << std::endl;

        std::cout << "slots of Mt: " << M_transposed->GetSlots() << std::endl;
        m_cc->Decrypt(M_transposed, sk, &debug);
        debug->SetLength(10);
        std::cout << "Mt: " << debug << std::endl;



        auto MM_transposed = this->eval_mult(m_M, M_transposed);
        

        m_cc->Decrypt(MM_transposed, sk, &debug);
        debug->SetLength(10);
        std::cout << "Level of MMt: " << MM_transposed->GetLevel() << std::endl;
        std::cout << "slots of MMt: " << MM_transposed->GetSlots() << std::endl;
        std::cout << "MMt: " << debug << std::endl;

        auto trace = this->eval_trace(MM_transposed, d*d);
        m_cc->Decrypt(trace, sk, &debug);
        std::cout << "Level of trace: " << trace->GetLevel() << std::endl;
        debug->SetLength(10);
        std::cout << "trace: " << debug << std::endl;

        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 5);

        m_cc->Decrypt(trace_reciprocal, sk, &debug);
        debug->SetLength(10);
        std::cout << "Level of MMt: " << trace_reciprocal->GetLevel() << std::endl;
        std::cout << "1/trace: " << debug << std::endl;

        Ciphertext<DCRTPoly> A_bar;
        Ciphertext<DCRTPoly> Y;

        Y =
            this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        std::cout << "slots of Y: " << Y->GetSlots() << std::endl;
        MM_transposed->SetSlots(this->max_batch);
        MM_transposed = this->clean(MM_transposed);
        A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                        MM_transposed, trace_reciprocal));
        std::cout << "slots of A: " << A_bar->GetSlots() << std::endl;

        std::cout << "level of Y: " << Y->GetLevel() << std::endl;
        std::cout << "level of A: " << A_bar->GetLevel() << std::endl;
        m_cc->Decrypt(Y, sk, &debug);
        debug->SetLength(10);
        std::cout << "Y: " << debug << std::endl;
        m_cc->Decrypt(A_bar, sk, &debug);
        debug->SetLength(8192);
        std::cout << "A_bar: " << debug << std::endl;

        for (int i = 0; i < this->r - 1; i++) {
          
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);   

            if ((int)Y->GetLevel() >= this->depth - 2) {
                Y = m_cc->EvalBootstrap(Y, 2, 18);     
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);   
            } 

            Y->SetSlots(this->max_batch);
            Y = this->clean(Y);
            A_bar->SetSlots(this->max_batch);
            A_bar = this->clean(A_bar);

            std::cout << "current interation: " << i << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl;
            std::cout << "level of A: " << A_bar->GetLevel() << std::endl;
            m_cc->Decrypt(Y, sk, &debug);
            debug->SetLength(8192);
            std::cout << "Y: " << debug << std::endl;
            m_cc->Decrypt(A_bar, sk, &debug);
            debug->SetLength(8192);
            std::cout << "A_bar: " << debug << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// Implementation using newCol matrix multiplication
template <int d>
class MatrixInverse_newColOpt : public MatrixInverseBase<d>,
                             public MatrixMult_newColOpt<d> {
  protected:
    using MatrixOperationBase<d>::rot;
    using MatrixOperationBase<d>::m_cc;
    using MatrixOperationBase<d>::vectorRotate;

  public:
    MatrixInverse_newColOpt(std::shared_ptr<Encryption> enc,
                         CryptoContext<DCRTPoly> cc,
                         PublicKey<DCRTPoly> publicKey)
        : MatrixInverseBase<d>(), MatrixMult_newColOpt<d>(enc, cc, publicKey) {}

    Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
        omp_set_max_active_levels(2);
        std::vector<double> vI = this->initializeIdentityMatrix();
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);
        Plaintext debug;

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        // auto trace = this->eval_trace(MM_transposed, d * d);
        // auto trace_reciprocal =
        //     this->m_cc->EvalDivide(trace, 1, (d * d) / 3 + d, 5);
        auto trace_reciprocal = 3.0 / (d*d);

        auto Y =
            this->m_cc->EvalMult(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMult(
                                        MM_transposed, trace_reciprocal));
        // auto Y =
        //     this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
        // auto A_bar =
        //     this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
        //                                 MM_transposed, trace_reciprocal));
       
        for (int i = 0; i < this->r - 1; i++) {
            auto identity = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d*d);
            auto A_plus_I = this->m_cc->EvalAdd(identity, A_bar);
            Y = this->eval_mult(Y, A_plus_I);
            A_bar = this->eval_mult(A_bar, A_bar);   

            // if ((int)Y->GetLevel() >= this->depth - 2) {
            //     A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
            //     Y = m_cc->EvalBootstrap(Y, 2, 18);
            // }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
    
    Ciphertext<DCRTPoly> eval_inverse_debug(const Ciphertext<DCRTPoly> &M, PrivateKey<DCRTPoly> sk){
        omp_set_max_active_levels(2);

        
        std::vector<double> vI = this->initializeIdentityMatrix();
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);
        Plaintext debug;

        auto M_transposed = this->eval_transpose(M);
        auto MM_transposed = this->eval_mult(M, M_transposed);

        m_cc->Decrypt(MM_transposed, sk, &debug);
        debug->SetLength(10);
        std::cout << "Level of MMt: " << MM_transposed->GetLevel() << std::endl;
        std::cout << "MMt: " << debug << std::endl;

        auto trace_reciprocal = 3.0 / (d*d);
        auto Y =
            this->m_cc->EvalMult(M_transposed, trace_reciprocal);
        auto A_bar =
            this->m_cc->EvalSub(pI, this->m_cc->EvalMult(
                                        MM_transposed, trace_reciprocal));
       
        for (int i = 0; i < this->r - 1; i++) {
            auto identity = m_cc->MakeCKKSPackedPlaintext(vI, 1, A_bar->GetLevel(), nullptr, d*d);
            auto A_plus_I = this->m_cc->EvalAdd(identity, A_bar);
            Y = this->eval_mult(Y, A_plus_I);
            A_bar = this->eval_mult(A_bar, A_bar);   

            std::cout << "current interation: " << i << std::endl;
            identity->SetLength(10);
            std::cout << "I: " << identity << std::endl;
            m_cc->Decrypt(A_plus_I, sk, &debug);
            debug->SetLength(10);
            std::cout << "A+I: " << debug << std::endl;
            m_cc->Decrypt(Y, sk, &debug);
            debug->SetLength(10);
            std::cout << "Y: " << debug << std::endl;
            m_cc->Decrypt(A_bar, sk, &debug);
            debug->SetLength(10);
            std::cout << "A_bar: " << debug << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl;
            std::cout << "level of A: " << A_bar->GetLevel() << std::endl;

            // if ((int)Y->GetLevel() >= this->depth - 2) {
            //     A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
            //     Y = m_cc->EvalBootstrap(Y, 2, 18);
            // }
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        std::cout << "level of Final Y: " << Y->GetLevel() << std::endl;
        return Y;
    }
};

