#pragma once

#include "matrix_algo_singlePack.h"
#include <memory>
#include <openfhe.h>

using namespace lbcrypto;

// Base class for matrix inverse operations
template <int d> class MatrixInverseBase {
  protected:
    static constexpr int r = 30;
    static constexpr int depth = 29;

    virtual std::vector<double> initializeIdentityMatrix(int batchSize) {
        std::vector<double> identity(batchSize, 0.0);
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
        std::vector<double> vI = this->initializeIdentityMatrix(this->max_batch);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, this->max_batch);

        auto m_M = M->Clone();
        m_M->SetSlots(this->max_batch);
        m_M = this->clean(m_M);

        auto M_transposed = this->eval_transpose(m_M, this->max_batch);
        auto MM_transposed = this->eval_mult(m_M, M_transposed);
        
        auto trace = this->eval_trace(MM_transposed, d * d);
        auto trace_reciprocal =
            this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 5);

        Ciphertext<DCRTPoly> A_bar;
        Ciphertext<DCRTPoly> Y;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                Y =
                    this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
            }
            #pragma omp section
            {
                MM_transposed->SetSlots(this->max_batch);
                MM_transposed = this->clean(MM_transposed);
                A_bar =
                    this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                                MM_transposed, trace_reciprocal));
            }
        }

        std::cout << "level of Y: " << Y->GetLevel() << std::endl;
        std::cout << "level of A: " << A_bar->GetLevel() << std::endl;

        for (int i = 0; i < this->r - 1; i++) {
            // #pragma omp parallel sections
            // {
            //     #pragma omp section
            //     {
            //         Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            //     }
            //     #pragma omp section
            //     {
            //         A_bar = this->eval_mult(A_bar, A_bar);   
            //     }
            // }

            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
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

        std::vector<double> vI = this->initializeIdentityMatrix(this->max_batch);
        Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI, 1, 0, nullptr, this->max_batch);

        auto m_M = M->Clone();
        m_M->SetSlots(this->max_batch);
        m_M = this->clean(m_M);
        auto M_transposed = this->eval_transpose(m_M, this->max_batch);

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

        auto trace = this->eval_trace(MM_transposed, d * d);
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

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                Y =
                    this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
                std::cout << "slots of Y: " << Y->GetSlots() << std::endl;

            }
            #pragma omp section
            {
                MM_transposed->SetSlots(this->max_batch);
                MM_transposed = this->clean(MM_transposed);
                A_bar =
                    this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
                                                MM_transposed, trace_reciprocal));
                std::cout << "slots of A: " << A_bar->GetSlots() << std::endl;
            }
        }

        std::cout << "level of Y: " << Y->GetLevel() << std::endl;
        std::cout << "level of A: " << A_bar->GetLevel() << std::endl;
        m_cc->Decrypt(Y, sk, &debug);
        debug->SetLength(10);
        std::cout << "Y: " << debug << std::endl;
        m_cc->Decrypt(A_bar, sk, &debug);
        debug->SetLength(10);
        std::cout << "A_bar: " << debug << std::endl;

        for (int i = 0; i < this->r - 1; i++) {
          
            Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
            A_bar = this->eval_mult(A_bar, A_bar);   

            if ((int)Y->GetLevel() >= this->depth - 2) {
                A_bar = m_cc->EvalBootstrap(A_bar, 2, 18);   
                Y = m_cc->EvalBootstrap(Y, 2, 18);     
            } 

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    // Y->SetSlots(this->max_batch);
                    Y = this->clean(Y);
                }
                #pragma omp section
                {
                    // A_bar->SetSlots(this->max_batch);
                    A_bar = this->clean(A_bar);
                }
            }
            std::cout << "current interation: " << i << std::endl;
            std::cout << "level of Y: " << Y->GetLevel() << std::endl;
            std::cout << "level of A: " << A_bar->GetLevel() << std::endl;
            m_cc->Decrypt(Y, sk, &debug);
            debug->SetLength(10);
            std::cout << "Y: " << debug << std::endl;
            m_cc->Decrypt(A_bar, sk, &debug);
            debug->SetLength(10);
            std::cout << "A_bar: " << debug << std::endl;
        }
        Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
        return Y;
    }
};

// Implementation using newCol matrix multiplication
// template <int d>
// class MatrixInverse_newCol : public MatrixInverseBase<d>,
//                              public MatrixMult_newCol<d> {
//   protected:
//     using MatrixOperationBase<d>::rot;
//     using MatrixOperationBase<d>::m_cc;
//     using MatrixOperationBase<d>::vectorRotate;

//   public:
//     MatrixInverse_newCol(std::shared_ptr<Encryption> enc,
//                          CryptoContext<DCRTPoly> cc,
//                          PublicKey<DCRTPoly> publicKey,
//                          std::vector<int> rotIndices, int iterations,
//                          int multDepth)
//         : MatrixInverseBase<d>(iterations, multDepth), MatrixMult_newCol<d>(
//                                                            enc, cc, publicKey,
//                                                            rotIndices) {}

//     Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly> &M) override {
//         std::vector<double> vI = this->initializeIdentityMatrix(d);
//         Plaintext pI = this->m_cc->MakeCKKSPackedPlaintext(vI);

//         auto M_transposed = this->eval_transpose(M);
//         auto MM_transposed = this->eval_mult(M, M_transposed);

//         auto trace = this->eval_trace(MM_transposed, d * d);
//         auto trace_reciprocal =
//             this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 50);

//         auto Y =
//             this->m_cc->EvalMultAndRelinearize(M_transposed, trace_reciprocal);
//         auto A_bar =
//             this->m_cc->EvalSub(pI, this->m_cc->EvalMultAndRelinearize(
//                                         MM_transposed, trace_reciprocal));

//         for (int i = 0; i < this->r - 1; i++) {
//             if (d >= 8 && (int)Y->GetLevel() >= this->depth - 2) {
//                 A_bar = m_cc->EvalBootstrap(A_bar, 2, 17);
//                 Y = m_cc->EvalBootstrap(Y, 2, 17);
//             }
//             Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
//             A_bar = this->eval_mult(A_bar, A_bar);
//         }
//         Y = this->eval_mult(Y, this->m_cc->EvalAdd(pI, A_bar));
//         return Y;
//     }
// };

