
#pragma once

#include <memory>
#include <openfhe.h>
#include <vector>

using namespace lbcrypto;

/*
 *   This code contains matrix algorithms
 *   where a matrix is encrypted into multiple ciphertexts.
 */

template <int d> // Matrix dimension d x d
class MatrixMultBase {
  protected:
    CryptoContext<DCRTPoly> m_cc;
    const Ciphertext<DCRTPoly> m_zeroCache;

    virtual Ciphertext<DCRTPoly> createZeroCache() {
        std::vector<double> zeroVec(N, 0.0);
        return m_enc->encryptInput(zeroVec);
    }

  public:
    MatrixMultBase(std::shared_ptr<Encryption> enc)
        : m_enc(enc), m_zeroCache(createZeroCache()) {}

    virtual ~MatrixMultBase() = default;
    virtual Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) = 0;

    virtual const Ciphertext<DCRTPoly> &getZero() const { return m_zeroCache; }
    constexpr size_t getMatrixSize() const { return d; }
};

// Secure Outsourced Matrix Computation and Application to Neural Networks, CCS
// 2018
template <int d> class MatrixMult_JKLS18 : public MatrixMultBase<d> {
  private:
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    RotationComposer<N> rot;

  public:
    std::shared_ptr<Encryption> m_enc;

    MatrixMult_JKLS18(CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> publicKey,
                      std::vector<int> rotIndices,
                      std::shared_ptr<Encryption> enc)
        : MatrixMultBase<d>(enc), m_cc(cc), m_PublicKey(publicKey),
          rot(m_cc, enc, rotIndices), m_enc(enc) {}

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {}
}

// On Matrix Multiplication with Homomorphic Encryption, CCSW 2022
template <int d>
class MatrixMult_RT22 : public MatrixMultBase<d> {
  private:
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    RotationComposer<N> rot;

  public:
    std::shared_ptr<Encryption> m_enc;

    MatrixMult_JKLS18(CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> publicKey,
                      std::vector<int> rotIndices,
                      std::shared_ptr<Encryption> enc)
        : MatrixMultBase<d>(enc), m_cc(cc), m_PublicKey(publicKey),
          rot(m_cc, enc, rotIndices), m_enc(enc) {}

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {}
}

// Secure and Efficient Outsourced Matrix Multiplication with Homomorphic
// Encryption, Indocrypt 2024
template <int d>
class MatrixMult_AS24 : public MatrixMultBase<d> {
  private:
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    RotationComposer<N> rot;

  public:
    std::shared_ptr<Encryption> m_enc;

    MatrixMult_JKLS18(CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> publicKey,
                      std::vector<int> rotIndices,
                      std::shared_ptr<Encryption> enc)
        : MatrixMultBase<d>(enc), m_cc(cc), m_PublicKey(publicKey),
          rot(m_cc, enc, rotIndices), m_enc(enc) {}

    Ciphertext<DCRTPoly>
    eval_mult(const Ciphertext<DCRTPoly> &matrixA,
              const Ciphertext<DCRTPoly> &matrixB) override {}
}