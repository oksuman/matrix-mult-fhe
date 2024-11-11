#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "matrix_inversion_algo.h"
#include "openfhe.h"
#include "rotation.h"

using namespace lbcrypto;

template <int d> class MatrixInverseNewColTestFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        switch (d) {
        case 4:
            r = 13;
            break;
        case 8:
            r = 16;
            break;
        case 16:
            r = 18;
            break;
        default:
            r = -1;
        }

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(2 * r + 12);
        parameters.SetScalingModSize(50);

        int max_batch = 1 << 16;
        int s = std::min(max_batch / d / d, d);
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
        for (int i = 1; i < d * d * s; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        m_cc->EvalRotateKeyGen(m_privateKey, rotations);
        m_cc->EvalMultKeyGen(m_privateKey);

        m_enc = std::make_shared<Encryption>(m_cc, m_publicKey);
        matInv = std::make_unique<MatrixInverse_newCol<d>>(
            m_enc, m_cc, m_publicKey, rotations, r);
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
        } while (!isInvertible(matrix));
        std::cout << "invertible matrix created" << std::endl;
        return matrix;
    }

    bool isInvertible(const std::vector<double> &matrix) {
        double max_elem = 0;
        double min_elem = std::numeric_limits<double>::max();
        for (size_t i = 0; i < d * d; i++) {
            max_elem = std::max(max_elem, std::abs(matrix[i]));
            min_elem = std::min(min_elem, std::abs(matrix[i]));
        }

        if (min_elem < 1e-6 || max_elem / min_elem > 1e6) {
            return false;
        }
        return true;
    }

    void verifyInverseResult(const std::vector<double> &original,
                             const std::vector<double> &inverse,
                             double threshold = 0.01) {
        std::vector<double> product(d * d, 0.0);
        bool failed = false;

        for (size_t i = 0; i < d; i++) {
            for (size_t j = 0; j < d; j++) {
                for (size_t k = 0; k < d; k++) {
                    product[i * d + j] +=
                        original[i * d + k] * inverse[k * d + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                if (std::abs(product[i * d + j] - expected) >= threshold) {
                    failed = true;
                }
                EXPECT_NEAR(product[i * d + j], expected, threshold)
                    << "Matrix multiplication failed at position (" << i << ","
                    << j << ")";
            }
        }

        if (failed) {
            printMatrix(original, "Original Matrix");
            printMatrix(inverse, "Inverse Matrix");
            printMatrix(product, "Product (should be identity)");
        }
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

    struct PrecisionMetrics {
        double max_error;
        double avg_log_precision;
        double min_log_precision;
        std::vector<std::pair<std::pair<int, int>, double>> worst_elements;
    };

    PrecisionMetrics analyzePrecision(const std::vector<double> &computed,
                                      const std::vector<double> &expected,
                                      int top_n = 5) {
        PrecisionMetrics metrics;
        metrics.max_error = 0.0;
        double sum_log_precision = 0.0;
        metrics.min_log_precision = std::numeric_limits<double>::infinity();

        std::vector<std::pair<double, std::pair<int, int>>> errors;

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double error =
                    std::abs(computed[i * d + j] - expected[i * d + j]);
                double log_precision =
                    error > 0 ? std::log2(error)
                              : std::numeric_limits<double>::infinity();

                metrics.max_error = std::max(metrics.max_error, error);
                if (log_precision < std::numeric_limits<double>::infinity()) {
                    sum_log_precision += log_precision;
                    metrics.min_log_precision =
                        std::min(metrics.min_log_precision, log_precision);
                }

                errors.push_back({error, {i, j}});
            }
        }

        std::sort(errors.begin(), errors.end(), std::greater<>());
        for (int i = 0; i < std::min(top_n, static_cast<int>(errors.size()));
             i++) {
            metrics.worst_elements.push_back(
                {errors[i].second, errors[i].first});
        }

        metrics.avg_log_precision = sum_log_precision / (d * d);
        return metrics;
    }

    std::vector<double> computeInverse(const std::vector<double> &matrix) {
        std::vector<double> result(d * d);
        std::vector<double> augmented(d * 2 * d);

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                augmented[i * (2 * d) + j] = matrix[i * d + j];
                augmented[i * (2 * d) + d + j] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < d; i++) {
            int pivot = i;
            double maxVal = std::abs(augmented[i * (2 * d) + i]);
            for (int j = i + 1; j < d; j++) {
                if (std::abs(augmented[j * (2 * d) + i]) > maxVal) {
                    maxVal = std::abs(augmented[j * (2 * d) + i]);
                    pivot = j;
                }
            }

            if (pivot != i) {
                for (int j = 0; j < 2 * d; j++) {
                    std::swap(augmented[i * (2 * d) + j],
                              augmented[pivot * (2 * d) + j]);
                }
            }

            double div = augmented[i * (2 * d) + i];
            for (int j = 0; j < 2 * d; j++) {
                augmented[i * (2 * d) + j] /= div;
            }

            for (int j = 0; j < d; j++) {
                if (j != i) {
                    double mult = augmented[j * (2 * d) + i];
                    for (int k = 0; k < 2 * d; k++) {
                        augmented[j * (2 * d) + k] -=
                            mult * augmented[i * (2 * d) + k];
                    }
                }
            }
        }

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                result[i * d + j] = augmented[i * (2 * d) + d + j];
            }
        }

        return result;
    }

    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_publicKey;
    PrivateKey<DCRTPoly> m_privateKey;
    std::shared_ptr<Encryption> m_enc;
    std::unique_ptr<MatrixInverse_newCol<d>> matInv;
    int r;
};

template <typename T>
class MatrixInverseNewColTestTyped
    : public MatrixInverseNewColTestFixture<T::value> {};

TYPED_TEST_SUITE_P(MatrixInverseNewColTestTyped);

TYPED_TEST_P(MatrixInverseNewColTestTyped, ComprehensiveInverseTest) {
    constexpr int d = TypeParam::value;
    std::cout << "\n=== Testing newCol " << d << "x" << d
              << " Matrix Inverse ===" << std::endl;

    auto matrix = this->generateRandomMatrix();
    auto enc_matrix = this->m_enc->encryptInput(matrix);
    auto inv_result = this->matInv->eval_inverse(enc_matrix);

    Plaintext result;
    this->m_cc->Decrypt(this->m_privateKey, inv_result, &result);
    result->SetLength(d * d);
    std::vector<double> computed_inverse = result->GetRealPackedValue();
    std::vector<double> expected_inverse = this->computeInverse(matrix);

    std::cout << "\nMultiplication Test Results:" << std::endl;
    this->verifyInverseResult(matrix, computed_inverse, 0.001);

    std::cout << "\nPrecision Analysis Results:" << std::endl;
    auto metrics = this->analyzePrecision(computed_inverse, expected_inverse);
    std::cout << "Maximum Error: " << metrics.max_error << std::endl;
    std::cout << "Average Log2 Precision: " << metrics.avg_log_precision
              << std::endl;
    std::cout << "Minimum Log2 Precision: " << metrics.min_log_precision
              << std::endl;
    std::cout << "\nWorst Elements:" << std::endl;
    for (const auto &elem : metrics.worst_elements) {
        std::cout << "Position (" << elem.first.first << ","
                  << elem.first.second << "): Error = " << elem.second
                  << std::endl;
    }

    EXPECT_LE(metrics.max_error, 0.0001);
    EXPECT_LT(metrics.avg_log_precision, -10.0);

    std::cout << "All tests completed for newCol " << d << "x" << d
              << " matrix\n"
              << std::endl;
}

REGISTER_TYPED_TEST_SUITE_P(MatrixInverseNewColTestTyped,
                            ComprehensiveInverseTest);

using InverseTestSizes = ::testing::Types<std::integral_constant<size_t, 4>,
                                          std::integral_constant<size_t, 8>,
                                          std::integral_constant<size_t, 16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(MatrixInverseNewCol,
                               MatrixInverseNewColTestTyped, InverseTestSizes);