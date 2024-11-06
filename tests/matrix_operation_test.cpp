#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d> class MatrixOperationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2);
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
        matOp = std::make_unique<TestMatrixOperation<d>>(
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

    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;
    PrivateKey<DCRTPoly> m_privateKey;
    std::shared_ptr<Encryption> m_enc;
    std::unique_ptr<TestMatrixOperation<d>> matOp;
};

template <typename T>
class MatrixOperationTestFixture : public MatrixOperationTest<T::value> {};

TYPED_TEST_SUITE_P(MatrixOperationTestFixture);

TYPED_TEST_P(MatrixOperationTestFixture, TransposeTest) {
    constexpr size_t d = TypeParam::value;

    auto matrix = this->generateRandomMatrix();

    // Create expected transpose result
    std::vector<double> expected(d * d);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            expected[j * d + i] = matrix[i * d + j];
        }
    }

    auto enc_matrix = this->m_enc->encryptInput(matrix);
    auto transpose_result = this->matOp->eval_transpose(enc_matrix);

    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, transpose_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    for (size_t i = 0; i < d * d; i++) {
        EXPECT_NEAR(decrypted[i], expected[i], 0.0001)
            << "Transpose mismatch at index " << i;
    }
}

TYPED_TEST_P(MatrixOperationTestFixture, TraceTest) {
    constexpr size_t d = TypeParam::value;

    auto matrix = this->generateRandomMatrix();

    // Calculate expected trace
    double expected_trace = 0.0;
    for (size_t i = 0; i < d; i++) {
        expected_trace += matrix[i * d + i];
    }

    auto enc_matrix = this->m_enc->encryptInput(matrix);
    auto trace_result = this->matOp->eval_trace(enc_matrix, d * d);

    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, trace_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    EXPECT_NEAR(decrypted[0], expected_trace, 0.0001) << "Trace value mismatch";
}

REGISTER_TYPED_TEST_SUITE_P(MatrixOperationTestFixture, TransposeTest,
                            TraceTest);

using TestSizes = ::testing::Types<
    std::integral_constant<size_t, 4>, std::integral_constant<size_t, 8>,
    std::integral_constant<size_t, 16>, std::integral_constant<size_t, 32>,
    std::integral_constant<size_t, 64>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixOperation, MatrixOperationTestFixture,
                               TestSizes);