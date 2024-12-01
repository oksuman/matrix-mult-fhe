#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "openfhe.h"

using namespace lbcrypto;

template <typename T>
class MatrixMultNewColTest : public ::testing::Test {
protected:
    static constexpr size_t d = T::value;

    void SetUp() override {
        // int max_batch = 1 << 16;

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2);
        parameters.SetScalingModSize(50);
        parameters.SetRingDim(1<<17);
        parameters.SetBatchSize(d * d);
        parameters.SetSecurityLevel(HEStd_NotSet);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);

        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;

        // 32768, -16384, -8192, -4096, -2048, -1024, -512, -256, -128, 
        std::vector<int> rotations={-64, -32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 64, 128, 192, 256, 320, 384, 448, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192, 16384, 32768};
        // std::vector<int> rotations={-64, -48, -32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 48, 56, 64, 128, 192, 256, 320, 384, 448, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192, 16384, 32768};
        // for (int i = 1; i < max_batch; i *= 2) {
        //     rotations.push_back(i);
        //     rotations.push_back(-i);
        // }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matMult = std::make_unique<MatrixMult_newColOpt<d>>(
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
    std::unique_ptr<MatrixMult_newColOpt<d>> matMult;
};

using TestSizes = ::testing::Types<std::integral_constant<size_t, 64>>;
TYPED_TEST_SUITE(MatrixMultNewColTest, TestSizes);

TYPED_TEST(MatrixMultNewColTest, GeneralMultiplicationTest) {
    auto matrixA = this->generateRandomMatrix();
    auto matrixB = this->generateRandomMatrix();

    constexpr size_t d = TestFixture::d;
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