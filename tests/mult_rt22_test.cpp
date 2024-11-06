#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <int d> class MatrixMultRT22Test : public ::testing::Test {
  protected:
    void SetUp() override {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2);
        parameters.SetScalingModSize(50);
        // parameters.SetRingDimension(1<<13);
        parameters.SetBatchSize(d * d * d);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;

        std::vector<int> rotations;
        for (int i = 1; i < d * d * d; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matMult = std::make_unique<MatrixMult_RT22<d>>(m_enc, m_cc, m_publicKey,
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

    std::vector<double> generateLargeRandomMatrix() { // 32x32 matrix generation
        std::vector<double> matrix(4 * d * d);        // 4 times larger than d*d
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int i = 0; i < 4 * d * d; i++) {
            matrix[i] = dis(gen);
        }
        return matrix;
    }

    std::vector<std::vector<double>>
    splitMatrix(const std::vector<double> &original) {
        // For d=16, we're splitting a 32x32 matrix into four 16x16 matrices
        std::vector<std::vector<double>> matrices(
            4, std::vector<double>(d * d, 0.0)); // each block is dxd (16x16)
        int largeSize = 2 * d;                   // 32 for 32x32 matrix

        for (int i = 0; i < d; ++i) {     // d rows per block
            for (int j = 0; j < d; ++j) { // d columns per block
                matrices[0][i * d + j] =
                    original[i * largeSize + j]; // top-left
                matrices[1][i * d + j] =
                    original[i * largeSize + (j + d)]; // top-right
                matrices[2][i * d + j] =
                    original[(i + d) * largeSize + j]; // bottom-left
                matrices[3][i * d + j] =
                    original[(i + d) * largeSize + (j + d)]; // bottom-right
            }
        }
        return matrices;
    }

    std::vector<double>
    mergeMatrices(const std::vector<std::vector<double>> &matrices) {
        // For d=16, we're merging four 16x16 matrices into one 32x32 matrix
        int largeSize = 2 * d; // 32 for resulting 32x32 matrix
        std::vector<double> mergedMatrix(largeSize * largeSize,
                                         0.0); // 32x32 matrix

        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                mergedMatrix[i * largeSize + j] =
                    matrices[0][i * d + j]; // top-left
                mergedMatrix[i * largeSize + (j + d)] =
                    matrices[1][i * d + j]; // top-right
                mergedMatrix[(i + d) * largeSize + j] =
                    matrices[2][i * d + j]; // bottom-left
                mergedMatrix[(i + d) * largeSize + (j + d)] =
                    matrices[3][i * d + j]; // bottom-right
            }
        }
        return mergedMatrix;
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

    std::vector<double>
    computeLargeExpectedProduct(const std::vector<double> &matrixA,
                                const std::vector<double> &matrixB) {
        int largeSize = 2 * d; // 32 for 32x32 matrix
        std::vector<double> result(largeSize * largeSize, 0.0);
        for (int i = 0; i < largeSize; i++) {
            for (int j = 0; j < largeSize; j++) {
                for (int k = 0; k < largeSize; k++) {
                    result[i * largeSize + j] +=
                        matrixA[i * largeSize + k] * matrixB[k * largeSize + j];
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
    std::unique_ptr<MatrixMult_RT22<d>> matMult;
};

template <typename T>
class MatrixMultRT22TestFixture : public MatrixMultRT22Test<T::value> {};

TYPED_TEST_SUITE_P(MatrixMultRT22TestFixture);

TYPED_TEST_P(MatrixMultRT22TestFixture, RegularMultiplicationTest) {
    constexpr int d = TypeParam::value;
    if (d >= 32)
        return; // Regular multiplication only for d < 32

    // Generate random matrices
    auto matrixA = this->generateRandomMatrix();
    auto matrixB = this->generateRandomMatrix();

    // Calculate expected result
    auto expected = this->computeExpectedProduct(matrixA, matrixB);

    // Encrypt matrices
    auto enc_matrixA = this->m_enc->encryptInput(matrixA);
    auto enc_matrixB = this->m_enc->encryptInput(matrixB);

    // Perform multiplication
    auto mult_result = this->matMult->eval_mult(enc_matrixA, enc_matrixB);

    // Decrypt result
    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, mult_result, &result);
    result->SetLength(d * d);
    std::vector<double> decrypted = result->GetRealPackedValue();

    // Check for errors and print debug info if needed
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

    // Compare results
    for (int i = 0; i < d * d; i++) {
        EXPECT_NEAR(decrypted[i], expected[i], 0.0001)
            << "Multiplication mismatch at index " << i;
    }
}

TYPED_TEST_P(MatrixMultRT22TestFixture, StrassenMultiplicationTest) {
    constexpr int d = TypeParam::value;
    if (d != 16)
        return;

    auto matrixA = this->generateLargeRandomMatrix();
    auto matrixB = this->generateLargeRandomMatrix();

    auto expected = this->computeLargeExpectedProduct(matrixA, matrixB);

    auto splitA = this->splitMatrix(matrixA);
    auto splitB = this->splitMatrix(matrixB);

    std::vector<Ciphertext<DCRTPoly>> enc_splitA;
    std::vector<Ciphertext<DCRTPoly>> enc_splitB;
    for (int i = 0; i < 4; i++) {
        enc_splitA.push_back(this->m_enc->encryptInput(splitA[i]));
        enc_splitB.push_back(this->m_enc->encryptInput(splitB[i]));
    }

    auto enc_result = this->matMult->eval_mult_strassen(enc_splitA, enc_splitB);

    std::vector<std::vector<double>> dec_splits;
    for (const auto &enc : enc_result) {
        Plaintext result;
        this->m_cc->Decrypt(this->m_privateKey, enc, &result);
        result->SetLength(d * d);
        dec_splits.push_back(result->GetRealPackedValue());
    }
    auto decrypted = this->mergeMatrices(dec_splits);

    int largeSize = 2 * d;
    for (int i = 0; i < largeSize * largeSize; i++) {
        EXPECT_NEAR(decrypted[i], expected[i], 0.0001)
            << "Strassen multiplication mismatch at index " << i;
    }
}

REGISTER_TYPED_TEST_SUITE_P(MatrixMultRT22TestFixture,
                            RegularMultiplicationTest,
                            StrassenMultiplicationTest);

using TestSizes = ::testing::Types<std::integral_constant<int, 4>,
                                   std::integral_constant<int, 8>,
                                   std::integral_constant<int, 16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixMultRT22, MatrixMultRT22TestFixture,
                               TestSizes);
