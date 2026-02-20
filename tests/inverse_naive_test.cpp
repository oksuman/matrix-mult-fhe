#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "naive_inversion.h"
#include "matrix_utils.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d>
class NaiveMatrixInverseTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Configure iteration count based on matrix size
        switch (d) {
        case 4:
            r = 18;
            break;
        case 8:
            r = 22;
            break;
        case 16:
            r = 25;
            break;
        default:
            r = -1;
        }

        // Set up crypto context
        CCParams<CryptoContextCKKSRNS> parameters;
        int multDepth = r + 9;
        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetScalingModSize(50);
        parameters.SetBatchSize(1); 
        parameters.SetSecurityLevel(HEStd_NotSet);
        parameters.SetRingDim(1<<4);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);
        m_cc->Enable(ADVANCEDSHE);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matOp = std::make_unique<MatrixOperations<d>>(m_enc, m_cc, m_publicKey);
    }


    std::vector<double> generateRandomMatrix() {
        std::vector<double> matrix(d * d);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        do {
            for (size_t i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));

        std::cout << "Invertible matrix generated!" << std::endl;
        return matrix;
    }


    std::vector<Ciphertext<DCRTPoly>> encryptMatrix(const std::vector<double>& matrix) {
        std::vector<Ciphertext<DCRTPoly>> encrypted(d * d);
        for(int i = 0; i < d * d; i++) {
            std::vector<double> value = {matrix[i]};
            encrypted[i] = m_enc->encryptInput(value);
        }
        return encrypted;
    }

    std::vector<double> decryptMatrix(const std::vector<Ciphertext<DCRTPoly>>& encrypted) {
        std::vector<double> decrypted(d * d);
        for(int i = 0; i < d * d; i++) {
            Plaintext result;
            m_cc->Decrypt(m_privateKey, encrypted[i], &result);
            decrypted[i] = result->GetRealPackedValue()[0];
        }
        return decrypted;
    }

    void verifyInverseResult(const std::vector<double>& original,
                            const std::vector<double>& inverse,
                            double threshold = 0.01) {
        std::vector<double> product(d * d, 0.0);
        bool failed = false;

        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                for (size_t k = 0; k < d; k++) {
                    product[i * d + j] += original[i * d + k] * inverse[k * d + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                if (std::abs(product[i * d + j] - expected) >= threshold) {
                    failed = true;
                }
                EXPECT_NEAR(product[i * d + j], expected, threshold)
                    << "Matrix multiplication failed at position (" << i << "," << j << ")";
            }
        }

        if (failed) {
            printMatrix(original, "Original Matrix");
            printMatrix(inverse, "Inverse Matrix");
            printMatrix(product, "Product (should be identity)");
        }
    }

    void printMatrix(const std::vector<double>& matrix, const std::string& name) {
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
    std::unique_ptr<MatrixOperations<d>> matOp;
    int r;
};

template <typename T>
class NaiveMatrixInverseTestTyped 
    : public NaiveMatrixInverseTestFixture<T::value> {};

TYPED_TEST_SUITE_P(NaiveMatrixInverseTestTyped);

TYPED_TEST_P(NaiveMatrixInverseTestTyped, ComprehensiveInverseTest) {
    constexpr int d = TypeParam::value;
    std::cout << "\n=== Testing " << d << "x" << d << " Matrix Inverse (Naive) ===" << std::endl;

    auto matrix = this->generateRandomMatrix();
    auto enc_matrix = this->encryptMatrix(matrix);

    auto inv_result = this->matOp->inverseMatrix(enc_matrix, this->r);
    std::cout << "Final Level: " << inv_result[0]->GetLevel() << std::endl;
    auto computed_inverse = this->decryptMatrix(inv_result);

    std::cout << "\nMultiplication Test Results:" << std::endl;
    this->verifyInverseResult(matrix, computed_inverse, 0.001);

    // Identity matrix test
    std::cout << "\nTesting Identity Matrix:" << std::endl;
    std::vector<double> identity(d * d, 0.0);
    for (int i = 0; i < d; i++) {
        identity[i * d + i] = 1.0;
    }

    auto enc_identity = this->encryptMatrix(identity);
    auto id_inv_result = this->matOp->inverseMatrix(enc_identity, this->r);
    auto computed_id_inverse = this->decryptMatrix(id_inv_result);

    this->verifyInverseResult(identity, computed_id_inverse, 0.001);
    
    std::cout << "All tests completed for " << d << "x" << d << " matrix\n" << std::endl;
}

REGISTER_TYPED_TEST_SUITE_P(NaiveMatrixInverseTestTyped, ComprehensiveInverseTest);

using InverseTestSizes = ::testing::Types<
    std::integral_constant<size_t, 4>,
    std::integral_constant<size_t, 8>,
    std::integral_constant<size_t, 16>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(NaiveMatrixInverse, 
                              NaiveMatrixInverseTestTyped,
                              InverseTestSizes);