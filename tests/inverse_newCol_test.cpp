#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <iomanip>
#include <numeric>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "openfhe.h"
#include "rotation.h"

using namespace lbcrypto;

template <int d>
class MatrixInverseNewColTestFixture : public ::testing::Test {
protected:
    struct ErrorStats {
        double max_error;
        double min_error;
        double avg_error;
        std::vector<std::tuple<int, int, double, double, double>> top_errors;
    };

    void SetUp() override {
        int multDepth = 34;
        uint32_t scaleModSize = 59;
        uint32_t firstModSize = 60;
        std::vector<uint32_t> levelBudget = {5, 5};
        std::vector<uint32_t> bsgsDim = {0, 0};
        CCParams<CryptoContextCKKSRNS> parameters;
        r = 30;
        int batchSize = 64*64;

        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetFirstModSize(firstModSize);
        parameters.SetScalingModSize(scaleModSize);
        parameters.SetBatchSize(batchSize);
        parameters.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(parameters);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);
        m_cc->Enable(ADVANCEDSHE);
        m_cc->Enable(FHE);

        std::vector<int> rotations = {-4032, -3528, -3024, -2520, -2016, -1512, -1008, -504, -64, -32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 63, 64, 126, 128, 189, 192, 252, 256, 315, 320, 378, 384, 441, 448, 504, 512, 1008, 1024, 1512, 1536, 2016, 2048, 2520, 2560, 3024, 3072, 3528, 3584, 4096, 8192, 16384, 32768};
        auto keyPair = m_cc->KeyGen();
        m_publicKey = keyPair.publicKey;
        m_privateKey = keyPair.secretKey;
        m_cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize);
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);
        m_cc->EvalBootstrapKeyGen(m_privateKey, batchSize);
        
        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matInv = std::make_unique<MatrixInverse_newColOpt<d>>(m_enc, m_cc, m_publicKey);
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
        return matrix;
    }

    std::vector<double> computeInverse(const std::vector<double>& matrix) {
        std::vector<std::vector<double>> mat(d, std::vector<double>(2 * d));
        std::vector<double> result(d * d);
        
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                mat[i][j] = matrix[i * d + j];
            }
            for (int j = d; j < 2 * d; j++) {
                mat[i][j] = (j - d == i) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < d; i++) {
            int pivot = i;
            double maxVal = std::abs(mat[i][i]);
            
            for (int j = i + 1; j < d; j++) {
                if (std::abs(mat[j][i]) > maxVal) {
                    maxVal = std::abs(mat[j][i]);
                    pivot = j;
                }
            }
            
            if (maxVal < 1e-10) {
                throw std::runtime_error("Matrix is singular");
            }

            if (pivot != i) {
                std::swap(mat[i], mat[pivot]);
            }

            double div = mat[i][i];
            for (int j = i; j < 2 * d; j++) {
                mat[i][j] /= div;
            }

            for (int j = 0; j < d; j++) {
                if (j != i) {
                    double factor = mat[j][i];
                    for (int k = i; k < 2 * d; k++) {
                        mat[j][k] -= factor * mat[i][k];
                    }
                }
            }
        }

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                result[i * d + j] = mat[i][j + d];
            }
        }

        return result;
    }

    ErrorStats analyzeErrors(const std::vector<double>& computed, 
                           const std::vector<double>& expected) {
        std::vector<std::tuple<int, int, double, double, double>> all_errors;
        ErrorStats stats;
        stats.max_error = 0.0;
        stats.min_error = std::numeric_limits<double>::max();
        double sum_error = 0.0;

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double error = std::abs(computed[i * d + j] - expected[i * d + j]);
                sum_error += error;
                stats.max_error = std::max(stats.max_error, error);
                stats.min_error = std::min(stats.min_error, error);
                all_errors.push_back(std::make_tuple(i, j, error, expected[i * d + j], computed[i * d + j]));
            }
        }

        stats.avg_error = sum_error / (d * d);

        // Sort errors in descending order and get top 5
        std::sort(all_errors.begin(), all_errors.end(),
                 [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

        stats.top_errors.assign(all_errors.begin(), 
                              all_errors.begin() + std::min(5, static_cast<int>(all_errors.size())));

        return stats;
    }

    void printErrorStats(const ErrorStats& stats) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Error Statistics:\n";
        std::cout << "Maximum Error: " << stats.max_error << "\n";
        std::cout << "Minimum Error: " << stats.min_error << "\n";
        std::cout << "Average Error: " << stats.avg_error << "\n\n";
        
        std::cout << "Top 5 Errors:\n";
        for (const auto& [i, j, error, expected, computed] : stats.top_errors) {
            std::cout << "Position (" << i << "," << j << "):\n";
            std::cout << "  Error: " << error << "\n";
            std::cout << "  Expected: " << expected << "\n";
            std::cout << "  Computed: " << computed << "\n";
        }
        std::cout << std::endl;
    }

    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;
    PrivateKey<DCRTPoly> m_privateKey;
    std::shared_ptr<Encryption> m_enc;
    std::unique_ptr<MatrixInverse_newColOpt<d>> matInv;
    int r;
};

template <typename T>
class MatrixInverseNewColTestTyped : public MatrixInverseNewColTestFixture<T::value> {};

TYPED_TEST_SUITE_P(MatrixInverseNewColTestTyped);

TYPED_TEST_P(MatrixInverseNewColTestTyped, InverseTest) {
    constexpr int d = TypeParam::value;
    
    auto matrix = this->generateRandomMatrix();
    auto enc_matrix = this->m_enc->encryptInput(matrix);
    auto inv_result = this->matInv->eval_inverse(enc_matrix);
    
    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, inv_result, &result);
    result->SetLength(d * d);
    
    std::vector<double> computed_inverse = result->GetRealPackedValue();
    std::vector<double> expected_inverse = this->computeInverse(matrix);
    
    auto error_stats = this->analyzeErrors(computed_inverse, expected_inverse);
    this->printErrorStats(error_stats);
    
    EXPECT_LE(error_stats.max_error, 0.0001) << "Matrix inverse error exceeds threshold";
}

REGISTER_TYPED_TEST_SUITE_P(MatrixInverseNewColTestTyped, InverseTest);

using InverseTestSizes = ::testing::Types<std::integral_constant<size_t, 64>>;
INSTANTIATE_TYPED_TEST_SUITE_P(MatrixInverseNewCol, MatrixInverseNewColTestTyped, InverseTestSizes);