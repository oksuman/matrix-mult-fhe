#pragma once

#include "encryption.h"
#include <openfhe.h>
#include <vector>
#include <memory>

using namespace lbcrypto;

template <int d>
class MatrixOperations {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;

public:
    MatrixOperations(std::shared_ptr<Encryption> enc,
                    CryptoContext<DCRTPoly> cc,
                    PublicKey<DCRTPoly> publicKey)
        : m_enc(enc), m_cc(cc), m_publicKey(publicKey) {}

    std::vector<Ciphertext<DCRTPoly>> 
    matrixMultiply(const std::vector<Ciphertext<DCRTPoly>>& A,
                   const std::vector<Ciphertext<DCRTPoly>>& B) {
        std::vector<Ciphertext<DCRTPoly>> C(d * d);
        
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                C[i*d + j] = m_cc->EvalMultAndRelinearize(
                    A[i*d], B[j]
                );
                
                for(int k = 1; k < d; k++) {
                    auto temp = m_cc->EvalMultAndRelinearize(
                        A[i*d + k], B[k*d + j]
                    );
                    m_cc->EvalAddInPlace(C[i*d + j], temp);
                }
            }
        }
        return C;
    }

    std::vector<Ciphertext<DCRTPoly>> 
    transpose(const std::vector<Ciphertext<DCRTPoly>>& M) {
        std::vector<Ciphertext<DCRTPoly>> result(d * d);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                result[j*d + i] = M[i*d + j]->Clone();
            }
        }
        return result;
    }

    Ciphertext<DCRTPoly> 
    trace(const std::vector<Ciphertext<DCRTPoly>>& M) {
        auto result = M[0]->Clone();
        for(int i = 1; i < d; i++) {
            m_cc->EvalAddInPlace(result, M[i*d + i]);
        }
        return result;
    }

    std::vector<Plaintext> createIdentityMatrix() {
        std::vector<Plaintext> I(d * d);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                std::vector<double> value(1, i == j ? 1.0 : 0.0);
                I[i*d + j] = m_cc->MakeCKKSPackedPlaintext(value);
            }
        }
        return I;
    }

    std::vector<Ciphertext<DCRTPoly>>
    matrixAddPlain(const std::vector<Ciphertext<DCRTPoly>>& A,
                   const std::vector<Plaintext>& B) {
        std::vector<Ciphertext<DCRTPoly>> C(d * d);
        for(int i = 0; i < d * d; i++) {
            // Copy plaintext to avoid const reference issue with OpenFHE's EvalAdd
            Plaintext ptx = B[i];
            C[i] = m_cc->EvalAdd(A[i], ptx);
        }
        return C;
    }

    // Scalar inverse using Newton-Raphson iteration (batchSize=1)
    Ciphertext<DCRTPoly> eval_scalar_inverse(const Ciphertext<DCRTPoly>& t,
                                              double upperBound,
                                              int iterations) {
        double x0 = 1.0 / upperBound;
        std::vector<double> x0_vec(1, x0);
        auto x = m_enc->encryptInput(x0_vec);
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        for (int i = 0; i < iterations; i++) {
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }
        return x;
    }

    std::vector<Ciphertext<DCRTPoly>>
    inverseMatrix(const std::vector<Ciphertext<DCRTPoly>>& M, int iterations, int scalar_inv_iter = 2) {
        auto M_transposed = transpose(M);
        auto MM_transposed = matrixMultiply(M, M_transposed);
        auto tr = trace(MM_transposed);
        auto trace_reciprocal = eval_scalar_inverse(tr, d * d, scalar_inv_iter);
        
        std::vector<Ciphertext<DCRTPoly>> Y(d * d);
        for(int i = 0; i < d * d; i++) {
            Y[i] = m_cc->EvalMultAndRelinearize(M_transposed[i], trace_reciprocal);
        }
        
        auto I = createIdentityMatrix();
        std::vector<Ciphertext<DCRTPoly>> A(d * d);
        for(int i = 0; i < d * d; i++) {
            auto temp = m_cc->EvalMultAndRelinearize(MM_transposed[i], trace_reciprocal);
            A[i] = m_cc->EvalSub(I[i], temp);
        }
        
        for(int iter = 0; iter < iterations; iter++) {
            auto I_plus_A = matrixAddPlain(A, I);
            Y = matrixMultiply(Y, I_plus_A);
            A = matrixMultiply(A, A);
        }
        
        return Y;
    }
};