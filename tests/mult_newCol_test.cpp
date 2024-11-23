#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d> class MatrixMultNewColTest : public ::testing::Test {
  protected:
    void SetUp() override {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2);
        parameters.SetScalingModSize(50);

        // For single matrix multiplication
        parameters.SetBatchSize(d * d);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);
        int max_batch = m_cc->GetRingDimension() / 2;
        std::cout << "ring dimension: " << m_cc->GetRingDimension()
                  << std::endl;
        int s = std::min(max_batch / d / d, d);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;

        std::vector<int> rotations;
        for (int i = 1; i < d * d * s; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matMult = std::make_unique<MatrixMult_newCol<d>>(
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
    std::unique_ptr<MatrixMult_newCol<d>> matMult;
};

// Define the test fixture
template <typename T>
class MatrixMultNewColTestFixture : public MatrixMultNewColTest<T::value> {};

// Register the test suite
TYPED_TEST_SUITE_P(MatrixMultNewColTestFixture);

// Define the test cases
TYPED_TEST_P(MatrixMultNewColTestFixture, IdentityMultiplicationTest) {
    constexpr size_t d = TypeParam::value;

    auto matrixA = this->generateRandomMatrix();

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

TYPED_TEST_P(MatrixMultNewColTestFixture, GeneralMultiplicationTest) {
    constexpr size_t d = TypeParam::value;

    auto matrixA = this->generateRandomMatrix();
    auto matrixB = this->generateRandomMatrix();

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

// Register all test cases
REGISTER_TYPED_TEST_SUITE_P(MatrixMultNewColTestFixture,
                            IdentityMultiplicationTest,
                            GeneralMultiplicationTest);

// Define the test sizes
using TestSizes = ::testing::Types<
    std::integral_constant<size_t, 4>, std::integral_constant<size_t, 8>,
    std::integral_constant<size_t, 16>, std::integral_constant<size_t, 32>,
    std::integral_constant<size_t, 64>>;

// Instantiate the test suite
INSTANTIATE_TYPED_TEST_SUITE_P(MatrixMultTests,             // Instance name
                               MatrixMultNewColTestFixture, // Fixture name
                               TestSizes);