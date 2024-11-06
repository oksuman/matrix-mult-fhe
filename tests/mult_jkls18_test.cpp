#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d> class MatrixMultJKLS18Test : public ::testing::Test {
  protected:
    void SetUp() override {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(3);
        parameters.SetScalingModSize(50);
        parameters.SetBatchSize(d * d);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;

        std::vector<int> rotations;
        for (int i = 1; i < d * d; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matMult = std::make_unique<MatrixMult_JKLS18<d>>(
            m_enc, m_cc, m_publicKey, rotations);
    }

    std::vector<double> generateRandomMatrix() {
        std::vector<double> matrix(d * d);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        for (size_t i = 0; i < d * d; i++) {
            matrix[i] = dis(gen);
        }
        return matrix;
    }

    // Helper function to print matrix for debugging
    void printMatrix(const std::vector<double> &matrix,
                     const std::string &name) {
        std::cout << name << ":\n";
        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                std::cout << matrix[i * d + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;
    PrivateKey<DCRTPoly> m_privateKey;
    std::shared_ptr<Encryption> m_enc;
    std::unique_ptr<MatrixMult_JKLS18<d>> matMult;
};

template <typename T>
class MatrixMultJKLS18TestFixture : public MatrixMultJKLS18Test<T::value> {};

TYPED_TEST_SUITE_P(MatrixMultJKLS18TestFixture);

TYPED_TEST_P(MatrixMultJKLS18TestFixture, IdentityMultiplicationTest) {
    constexpr size_t d = TypeParam::value;

    // Generate random matrix A
    auto matrixA = this->generateRandomMatrix();

    // Create identity matrix B
    std::vector<double> matrixB(d * d, 0.0);
    for (size_t i = 0; i < d; i++) {
        matrixB[i * d + i] = 1.0;
    }

    auto enc_matrixA = this->m_enc->encryptInput(matrixA);
    auto enc_matrixB = this->m_enc->encryptInput(matrixB);

    auto mult_result = this->matMult->eval_mult(enc_matrixA, enc_matrixB);
    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, mult_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    for (size_t i = 0; i < d * d; i++) {
        EXPECT_NEAR(decrypted[i], matrixA[i], 0.0001)
            << "Identity multiplication failed at index " << i;
    }
}

TYPED_TEST_P(MatrixMultJKLS18TestFixture, GeneralMultiplicationTest) {
    constexpr size_t d = TypeParam::value;

    auto matrixA = this->generateRandomMatrix();
    auto matrixB = this->generateRandomMatrix();

    // Calculate expected result
    std::vector<double> expected(d * d, 0.0);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            for (size_t k = 0; k < d; k++) {
                expected[i * d + j] += matrixA[i * d + k] * matrixB[k * d + j];
            }
        }
    }

    auto enc_matrixA = this->m_enc->encryptInput(matrixA);
    auto enc_matrixB = this->m_enc->encryptInput(matrixB);

    auto mult_result = this->matMult->eval_mult(enc_matrixA, enc_matrixB);
    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, mult_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    // Debug output for failed tests
    bool failed = false;
    for (size_t i = 0; i < d * d; i++) {
        if (std::abs(decrypted[i] - expected[i]) >= 0.0001) {
            failed = true;
            break;
        }
    }

    if (failed) {
        this->printMatrix(matrixA, "Matrix A");
        this->printMatrix(matrixB, "Matrix B");
        this->printMatrix(expected, "Expected Result");
        this->printMatrix(decrypted, "Actual Result");
    }

    for (size_t i = 0; i < d * d; i++) {
        EXPECT_NEAR(decrypted[i], expected[i], 0.0001)
            << "Multiplication mismatch at index " << i;
    }
}

REGISTER_TYPED_TEST_SUITE_P(MatrixMultJKLS18TestFixture,
                            IdentityMultiplicationTest,
                            GeneralMultiplicationTest);

using TestSizes = ::testing::Types<
    std::integral_constant<size_t, 4>, std::integral_constant<size_t, 8>,
    std::integral_constant<size_t, 16>, std::integral_constant<size_t, 32>,
    std::integral_constant<size_t, 64>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixMultJKLS18, MatrixMultJKLS18TestFixture,
                               TestSizes);