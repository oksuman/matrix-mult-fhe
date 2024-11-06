#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d> class MatrixMultAS24Test : public ::testing::Test {
  protected:
    void SetUp() override {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2);
        parameters.SetScalingModSize(50);

        // For single matrix multiplication
        int max_batch = 1 << 13;
        int s = std::min(max_batch / d / d, d);

        parameters.SetBatchSize(d * d * s);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;

        // Generate rotation keys for batch size
        std::vector<int> rotations;
        for (int i = 1; i < d * d * s; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matMult = std::make_unique<MatrixMult_AS24<d>>(m_enc, m_cc, m_publicKey,
                                                       rotations);
    }

    std::vector<double> generateRandomMatrix() {
        std::vector<double> matrix(d * d);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int i = 0; i < d * d; i++) {
            matrix[i] = dis(gen);
        }
        return matrix;
    }

    std::vector<double>
    computeExpectedProduct(const std::vector<double> &matrixA,
                           const std::vector<double> &matrixB) {
        std::vector<double> result(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                for (int k = 0; k < d; k++) {
                    result[i * d + j] +=
                        matrixA[i * d + k] * matrixB[k * d + j];
                }
            }
        }
        return result;
    }

    void printMatrix(const std::vector<double> &matrix,
                     const std::string &name) {
        std::cout << name << ":\n";
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
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
    std::unique_ptr<MatrixMult_AS24<d>> matMult;
};

template <typename T>
class MatrixMultAS24TestFixture : public MatrixMultAS24Test<T::value> {};

TYPED_TEST_SUITE_P(MatrixMultAS24TestFixture);

TYPED_TEST_P(MatrixMultAS24TestFixture, MultiplicationTest) {
    constexpr int d = TypeParam::value;

    auto matrixA = this->generateRandomMatrix();
    auto matrixB = this->generateRandomMatrix();

    auto expected = this->computeExpectedProduct(matrixA, matrixB);

    auto enc_matrixA = this->m_enc->encryptInput(matrixA);
    auto enc_matrixB = this->m_enc->encryptInput(matrixB);

    auto mult_result = this->matMult->eval_mult(enc_matrixA, enc_matrixB);

    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, mult_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    bool failed = false;
    for (int i = 0; i < d * d; i++) {
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

    for (int i = 0; i < d * d; i++) {
        EXPECT_NEAR(decrypted[i], expected[i], 0.0001)
            << "Multiplication mismatch at index " << i;
    }
}

REGISTER_TYPED_TEST_SUITE_P(MatrixMultAS24TestFixture, MultiplicationTest);

using TestSizes = ::testing::Types<
    std::integral_constant<int, 4>, std::integral_constant<int, 8>,
    std::integral_constant<int, 16>, std::integral_constant<int, 32>,
    std::integral_constant<int, 64>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixMultAS24, MatrixMultAS24TestFixture,
                               TestSizes);