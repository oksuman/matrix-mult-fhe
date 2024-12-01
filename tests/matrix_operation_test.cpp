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
        parameters.SetMultiplicativeDepth(10);
        parameters.SetScalingModSize(50);
        parameters.SetBatchSize(d * d);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);
        m_cc->Enable(ADVANCEDSHE);

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
            m_enc, m_cc, m_publicKey);
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

TYPED_TEST_P(MatrixOperationTestFixture, TraceReciprocalTest) {
    constexpr size_t d = TypeParam::value;
    auto M = this->generateRandomMatrix();
    
    // Calculate M * M^T
    std::vector<double> MM_transposed(d * d, 0.0);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < d; k++) {
                sum += M[i * d + k] * M[j * d + k];
            }
            MM_transposed[i * d + j] = sum;
        }
    }
    
    // Calculate expected trace of MM^T
    double expected_trace = 0.0;
    for (size_t i = 0; i < d; i++) {
        expected_trace += MM_transposed[i * d + i];
    }
    
    // Calculate expected reciprocal
    double expected_reciprocal = 1.0 / expected_trace;
    
    // Encrypt and calculate encrypted result
    auto enc_M = this->m_enc->encryptInput(MM_transposed);
    auto trace = this->matOp->eval_trace(enc_M, d * d);
    auto trace_reciprocal = 
        this->m_cc->EvalDivide(trace, (d * d) / 3 - d, (d * d) / 3 + d, 5);
    
    std::cout << "Final Level: " << trace_reciprocal->GetLevel() << std::endl;
    
    // Decrypt result
    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, trace_reciprocal, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();
    
    // Calculate error statistics
    double max_error = 0.0;
    double total_error = 0.0;
    
    for (size_t i = 0; i < d * d; i++) {
        double error = std::abs(decrypted[i] - expected_reciprocal);
        max_error = std::max(max_error, error);
        total_error += error;
        
        EXPECT_NEAR(decrypted[i], expected_reciprocal, 0.001) 
            << "Trace reciprocal mismatch at index " << i;
        EXPECT_FALSE(std::isinf(decrypted[i])) 
            << "Result is infinite at index " << i;
        EXPECT_FALSE(std::isnan(decrypted[i])) 
            << "Result is NaN at index " << i;
    }
    
    double avg_error = total_error / (d * d);
    
    std::cout << "Error Statistics:" << std::endl;
    std::cout << "Maximum Error: " << max_error << std::endl;
    std::cout << "Average Error: " << avg_error << std::endl;
    std::cout << "Expected Value: " << expected_reciprocal << std::endl;
}

REGISTER_TYPED_TEST_SUITE_P(MatrixOperationTestFixture, TransposeTest, TraceTest, TraceReciprocalTest);

using TestSizes = ::testing::Types<std::integral_constant<size_t, 64>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixOperation, MatrixOperationTestFixture,
                               TestSizes);