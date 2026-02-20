#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "diagonal_packing.h"
#include "encryption.h"
#include "matrix_algo_multiPack.h"
#include "matrix_utils.h"
#include "openfhe.h"
#include "rotation.h"

using namespace lbcrypto;

template <int d> class MatrixInvDiagTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set r based on matrix dimension
        int r;
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
        case 32:
            r = 27;
            break;
        case 64:
            r = 30;
            break;
        default:
            r = -8; // For multiplication-only tests
        }

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(r + 9);
        parameters.SetScalingModSize(50);
        parameters.SetBatchSize(d);
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
        for (int i = 1; i < d; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matInv = std::make_unique<MatrixInv_diag<d>>(m_enc, m_cc, m_publicKey,
                                                     rotations, r);
    }

    std::vector<double> generateInvertibleMatrix() {
        std::vector<double> matrix(d * d);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        do {
            // Generate diagonal dominant matrix
            for (size_t i = 0; i < d; i++) {
                double sum = 0;
                for (size_t j = 0; j < d; j++) {
                    if (i != j) {
                        matrix[i * d + j] =
                            dis(gen) * 0.1; // Off-diagonal elements
                        sum += std::abs(matrix[i * d + j]);
                    }
                }
                matrix[i * d + i] = sum + dis(gen); // Make diagonal dominant
            }
        } while (!utils::isInvertible(matrix, d));

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
    std::unique_ptr<MatrixInv_diag<d>> matInv;
};

template <typename T>
class MatrixInvDiagTestFixture : public MatrixInvDiagTest<T::value> {};

TYPED_TEST_SUITE_P(MatrixInvDiagTestFixture);

TYPED_TEST_P(MatrixInvDiagTestFixture, MultTest) {
    constexpr size_t d = TypeParam::value;

    auto matrixA = this->generateInvertibleMatrix();
    auto matrixB = this->generateInvertibleMatrix();

    // Extract diagonals
    auto diagA = utils::extractDiagonalVectors(matrixA, d);
    auto diagB = utils::extractDiagonalVectors(matrixB, d);

    // Encrypt diagonals
    std::vector<Ciphertext<DCRTPoly>> encA, encB;
    for (const auto &diag : diagA) {
        encA.push_back(this->m_enc->encryptInput(diag));
    }
    for (const auto &diag : diagB) {
        encB.push_back(this->m_enc->encryptInput(diag));
    }

    // Perform multiplication
    auto mult_result = this->matInv->eval_mult(encA, encB);

    // Decrypt and pack results
    std::vector<std::vector<double>> decrypted_diags;
    for (const auto &cipher : mult_result) {
        Plaintext result;
        this->m_cc->Decrypt(this->m_privateKey, cipher, &result);
        result->SetLength(d);
        decrypted_diags.push_back(result->GetRealPackedValue());
    }
    auto decrypted = utils::packDiagonalVectors(decrypted_diags, d);

    // Calculate expected result
    std::vector<double> expected(d * d, 0.0);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            for (size_t k = 0; k < d; k++) {
                expected[i * d + j] += matrixA[i * d + k] * matrixB[k * d + j];
            }
        }
    }

    // Debug output for failed tests
    bool failed = false;
    for (size_t i = 0; i < d * d; i++) {
        if (std::abs(decrypted[i] - expected[i]) >= 0.00001) {
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
        EXPECT_NEAR(decrypted[i], expected[i], 0.00001)
            << "Multiplication mismatch at index " << i;
    }
}

TYPED_TEST_P(MatrixInvDiagTestFixture, InverseTest) {
    constexpr size_t d = TypeParam::value;

    if (d > 16) {
        GTEST_SKIP() << "Skipping inverse test for matrix size " << d;
        return;
    }

    auto matrix = this->generateInvertibleMatrix();

    // Extract diagonals and encrypt
    auto diags = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto &diag : diags) {
        enc_matrix.push_back(this->m_enc->encryptInput(diag));
    }

    // Calculate inverse
    auto inv_result = this->matInv->eval_inverse(enc_matrix);

    // Decrypt and pack results
    std::vector<std::vector<double>> decrypted_diags;
    for (const auto &cipher : inv_result) {
        Plaintext result;
        this->m_cc->Decrypt(this->m_privateKey, cipher, &result);
        result->SetLength(d);
        decrypted_diags.push_back(result->GetRealPackedValue());
    }
    auto decrypted = utils::packDiagonalVectors(decrypted_diags, d);

    // Verify result by multiplying with original matrix
    std::vector<double> identity(d * d, 0.0);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            for (size_t k = 0; k < d; k++) {
                identity[i * d + j] += matrix[i * d + k] * decrypted[k * d + j];
            }
        }
    }

    bool failed = false;
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(identity[i * d + j] - expected) >= 0.00001) {
                failed = true;
                break;
            }
        }
    }

    if (failed) {
        this->printMatrix(matrix, "Original Matrix");
        this->printMatrix(decrypted, "Inverse Matrix");
        this->printMatrix(identity, "Product (should be identity)");
    }

    // Check if result is close to identity matrix
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            if (i == j) {
                EXPECT_NEAR(identity[i * d + j], 1.0, 0.00001)
                    << "Diagonal element mismatch at (" << i << "," << j << ")";
            } else {
                EXPECT_NEAR(identity[i * d + j], 0.0, 0.00001)
                    << "Off-diagonal element mismatch at (" << i << "," << j
                    << ")";
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(MatrixInvDiagTestFixture, MultTest, InverseTest);

// using MultTestSizes = ::testing::Types<
//     std::integral_constant<size_t, 4>, std::integral_constant<size_t, 8>,
//     std::integral_constant<size_t, 16>, std::integral_constant<size_t, 32>,
//     std::integral_constant<size_t, 64>>;

using InvTestSizes = ::testing::Types<std::integral_constant<size_t, 4>,
                                      std::integral_constant<size_t, 8>,
                                      std::integral_constant<size_t, 16>>;

// INSTANTIATE_TYPED_TEST_SUITE_P(MatrixMultTests, MatrixInvDiagTestFixture,
//                                MultTestSizes);

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixInvTests, MatrixInvDiagTestFixture,
                               InvTestSizes);