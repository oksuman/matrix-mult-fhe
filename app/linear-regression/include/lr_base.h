// lr_base.h - Base class for encrypted linear regression
#pragma once

#include "encryption.h"
#include "rotation.h"
#include "csv_processor.h"
#include "mse_calculator.h"
#include <map>
#include <memory>
#include <chrono>
#include <tuple>

const int FEATURE_DIM = 8;
const int SAMPLE_DIM = 64;

// Unified iteration counts
const int SCALAR_INV_ITERATIONS = 1;

// Matrix inversion iterations by dimension (95th percentile)
inline int getInversionIterations(int d) {
    switch(d) {
        case 4:  return 18;
        case 8:  return 22;
        case 16: return 25;
        case 32: return 27;
        case 64: return 30;
        default: return 25;
    }
}

class LinearRegressionBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    std::vector<int> m_rotIndices;
    Ciphertext<DCRTPoly> m_weights;
    RotationComposer rot;
    mutable std::map<int, Ciphertext<DCRTPoly>> m_zeroCache;

    Ciphertext<DCRTPoly> makeZero(int batchSize) {
        auto it = m_zeroCache.find(batchSize);
        if (it == m_zeroCache.end()) {
            std::vector<double> zeroVec(batchSize, 0.0);
            auto zeroPtx = m_cc->MakeCKKSPackedPlaintext(zeroVec, 1, 0, nullptr, batchSize);
            m_zeroCache[batchSize] = m_cc->Encrypt(zeroPtx, m_keyPair.publicKey);
        }
        return m_zeroCache.at(batchSize)->Clone();
    }

    using TimingResult = std::tuple<std::chrono::duration<double>,
                                   std::chrono::duration<double>,
                                   std::chrono::duration<double>,
                                   std::chrono::duration<double>>;

    std::vector<double> vectorRotate(const std::vector<double> &vec, int rotateIndex) {
        if (vec.empty()) return std::vector<double>();
        std::vector<double> result = vec;
        int n = result.size();
        if (rotateIndex > 0) {
            std::rotate(result.begin(), result.begin() + rotateIndex, result.end());
        } else if (rotateIndex < 0) {
            rotateIndex += n;
            std::rotate(result.begin(), result.begin() + rotateIndex, result.end());
        }
        return result;
    }

    // ========== JKLS18 Matrix Multiplication ==========

    std::vector<double> generateSigmaMsk(int k, int d) {
        std::vector<double> u(d * d, 0);
        if (k >= 0) {
            for (int i = d * k; i < d - k + d * k && i < d * d; ++i) {
                u[i] = 1.0;
            }
        } else {
            for (int i = 0; i < d * d; i++) {
                if (i < d + d * (d + k) && i >= -k + d * (d + k))
                    u[i] = 1.0;
            }
        }
        return u;
    }

    std::vector<double> generateTauMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = k; i < d * d; i += d)
            msk[i] = 1;
        return msk;
    }

    std::vector<double> generateShiftingMsk(int k, int d) {
        std::vector<double> v(d * d, 0);
        for (int i = k; i < d * d; i += d) {
            for (int j = i; j < i + d - k; ++j) {
                v[j] = 1;
            }
        }
        return v;
    }

    Ciphertext<DCRTPoly> columnShifting(const Ciphertext<DCRTPoly>& M, int l, int d) {
        if (l == 0) return M->Clone();
        std::vector<double> msk = generateShiftingMsk(l, d);
        Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
        auto tmp = m_cc->EvalMult(pmsk, M);
        auto M_1 = rot.rotate(m_cc->EvalSub(M, tmp), l - d);
        auto M_2 = rot.rotate(tmp, l);
        return m_cc->EvalAdd(M_1, M_2);
    }

    Ciphertext<DCRTPoly> sigmaTransform(const Ciphertext<DCRTPoly>& M, int d) {
        auto sigma_M = makeZero(d * d);

        int bs = (int)round(sqrt((double)d));

        std::vector<Ciphertext<DCRTPoly>> babySteps(bs);
        for (int i = 0; i < bs; i++) {
            babySteps[i] = rot.rotate(M, i);
        }

        for (int i = 1; i < d - bs * (bs - 1); i++) {
            Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                generateSigmaMsk(-d + i, d), 1, 0, nullptr, d * d);
            m_cc->EvalAddInPlace(sigma_M,
                m_cc->EvalMult(rot.rotate(M, i - d), pmsk));
        }

        for (int i = -(bs - 1); i < bs; i++) {
            auto tmp = makeZero(d * d);
            for (int j = 0; j < bs; j++) {
                auto msk = generateSigmaMsk(bs * i + j, d);
                msk = vectorRotate(msk, -bs * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(pmsk, babySteps[j]));
            }
            m_cc->EvalAddInPlace(sigma_M, rot.rotate(tmp, bs * i));
        }

        return sigma_M;
    }

    Ciphertext<DCRTPoly> tauTransform(const Ciphertext<DCRTPoly>& M, int d) {
        auto tau_M = makeZero(d * d);

        double squareRootd = sqrt((double)d);
        int squareRootIntd = (int)squareRootd;

        if (squareRootIntd * squareRootIntd == d) {
            std::vector<Ciphertext<DCRTPoly>> babySteps(squareRootIntd);
            for (int i = 0; i < squareRootIntd; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < squareRootIntd; i++) {
                auto tmp = makeZero(d * d);
                for (int j = 0; j < squareRootIntd; j++) {
                    auto msk = generateTauMsk(squareRootIntd * i + j, d);
                    msk = vectorRotate(msk, -squareRootIntd * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                    m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, squareRootIntd * d * i));
            }
        } else {
            int steps = (int)round(squareRootd);

            std::vector<Ciphertext<DCRTPoly>> babySteps(steps);
            for (int i = 0; i < steps; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < d - steps * (steps - 1); i++) {
                Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                    generateTauMsk(steps * (steps - 1) + i, d), 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tau_M,
                    m_cc->EvalMult(rot.rotate(M, (steps * (steps - 1) + i) * d), pmsk));
            }

            for (int i = 0; i < steps - 1; i++) {
                auto tmp = makeZero(d * d);
                for (int j = 0; j < steps; j++) {
                    auto msk = generateTauMsk(steps * i + j, d);
                    msk = vectorRotate(msk, -steps * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                    m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, steps * d * i));
            }
        }

        return tau_M;
    }

    // JKLS18 d×d matrix multiplication: C = A * B
    Ciphertext<DCRTPoly> eval_mult_JKLS18(const Ciphertext<DCRTPoly>& A,
                                          const Ciphertext<DCRTPoly>& B, int d) {
        auto sigma_A = sigmaTransform(A, d);
        auto tau_B = tauTransform(B, d);
        auto matrixC = m_cc->EvalMultAndRelinearize(sigma_A, tau_B);

        for (int i = 1; i < d; i++) {
            auto shifted_A = columnShifting(sigma_A, i, d);
            tau_B = rot.rotate(tau_B, d);
            m_cc->EvalAddInPlace(matrixC,
                m_cc->EvalMultAndRelinearize(shifted_A, tau_B));
        }

        return matrixC;
    }

    // ========== Scalar Operations ==========

    // Power series scalar inverse: 1/t with upper bound constraint
    Ciphertext<DCRTPoly> eval_scalar_inverse(const Ciphertext<DCRTPoly>& t, double upperBound, int iterations, int batchSize) {
        double x0 = 1.0 / upperBound;
        auto x = m_cc->Encrypt(m_keyPair.publicKey,
            m_cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, x0), 1, 0, nullptr, batchSize));
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        for (int i = 0; i < iterations; i++) {
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }
        return x;
    }

    // Matrix trace
    Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        std::vector<double> msk(batchSize, 0);
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }
        auto trace = m_cc->EvalMult(M, m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, batchSize));

        for (int i = 1; i <= log2(batchSize); i++) {
            m_cc->EvalAddInPlace(trace, rot.rotate(trace, batchSize / (1 << i)));
        }
        return trace;
    }

    // ========== Transpose ==========

    std::vector<double> generateTransposeMask(int k, int d) {
        std::set<int> indices;
        if (k >= 0) {
            for (int j = 0; j < d - k; j++) {
                indices.insert((d + 1) * j + k);
            }
        } else {
            for (int j = -k; j < d; j++) {
                indices.insert((d + 1) * j + k);
            }
        }
        std::vector<double> msk(d * d, 0);
        for (int index : indices) {
            msk[index] = 1.0;
        }
        return msk;
    }

    // BSGS optimized transpose: O(2*sqrt(d)) rotations instead of O(2d)
    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        int bs = (int)round(sqrt((double)d));  // baby step size (e.g., 8 for d=64)

        // Baby steps: pre-compute bs rotations
        std::vector<Ciphertext<DCRTPoly>> babyStepsOfM(bs);
        for (int i = 0; i < bs; i++) {
            babyStepsOfM[i] = rot.rotate(M, (d - 1) * i);
        }

        // Zero ciphertext for accumulation
        auto M_transposed = makeZero(batchSize);

        // Giant steps: iterate over blocks of size bs
        for (int i = -bs; i < bs; i++) {
            auto tmp = makeZero(batchSize);

            int js = (i == -bs) ? 1 : 0;  // skip k=0 case when i=-bs (k=-d*bs would be out of range)

            for (int j = js; j < bs; j++) {
                int k = bs * i + j;  // diagonal index
                if (k >= d || k <= -d) continue;  // skip out-of-range diagonals

                auto vmsk = generateTransposeMask(k, d);
                vmsk = vectorRotate(vmsk, -bs * (d - 1) * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(vmsk, 1, 0, nullptr, batchSize);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babyStepsOfM[j], pmsk));
            }

            tmp = rot.rotate(tmp, bs * (d - 1) * i);
            m_cc->EvalAddInPlace(M_transposed, tmp);
        }

        return M_transposed;
    }

    Ciphertext<DCRTPoly> computeXty(const Ciphertext<DCRTPoly>& X,
                                   const Ciphertext<DCRTPoly>& y,
                                   int featureDim, int sampleDim) {
        auto y_replicated = y->Clone();
        y_replicated->SetSlots(sampleDim * sampleDim);

        auto y_transposed = eval_transpose(y_replicated, sampleDim, sampleDim * sampleDim);

        auto result = m_cc->EvalMultAndRelinearize(X, y_transposed);

        for (int i = 0; i < (int)log2(sampleDim); i++) {
            m_cc->EvalAddInPlace(result, rot.rotate(result, sampleDim * (1 << i)));
        }

        result->SetSlots(sampleDim);

        for (int i = 0; i < (int)log2(sampleDim / featureDim); i++) {
            m_cc->EvalAddInPlace(result, rot.rotate(result, featureDim * (1 << i)));
        }

        return result;
    }

    // ========== Identity Matrix ==========

    virtual std::vector<double> initializeIdentityMatrix(size_t dim) {
        std::vector<double> identity(dim * dim, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

    virtual std::vector<double> initializeIdentityMatrix2(size_t dim, int batchSize) {
        std::vector<double> identity(batchSize, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

    // Debug helper: decrypt and print matrix
    void debugPrintMatrix(const std::string& label, const Ciphertext<DCRTPoly>& cipher,
                         int rows, int cols, int stride = 0) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        auto vec = ptx->GetRealPackedValue();

        if (stride == 0) stride = cols;
        int showRows = std::min(rows, 8);
        int showCols = std::min(cols, 8);

        std::cout << "=== " << label << " (" << rows << "x" << cols
                  << ", slots=" << cipher->GetSlots() << ", level=" << cipher->GetLevel() << ") ===" << std::endl;
        for (int i = 0; i < showRows; i++) {
            std::cout << "  ";
            for (int j = 0; j < showCols; j++) {
                std::cout << std::setw(12) << std::setprecision(6) << std::fixed << vec[i * stride + j];
            }
            if (cols > 8) std::cout << " ...";
            std::cout << std::endl;
        }
        if (rows > 8) std::cout << "  ..." << std::endl;
        std::cout << std::endl;
    }

    void debugPrintVector(const std::string& label, const Ciphertext<DCRTPoly>& cipher, int len) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        auto vec = ptx->GetRealPackedValue();

        int show = std::min(len, 8);
        std::cout << "=== " << label << " (len=" << len << ", slots=" << cipher->GetSlots() << ") ===" << std::endl;
        std::cout << "  ";
        for (int i = 0; i < show; i++) {
            std::cout << std::setw(12) << std::setprecision(6) << std::fixed << vec[i];
        }
        if (len > 8) std::cout << " ...";
        std::cout << std::endl << std::endl;
    }

    double debugGetTrace(const Ciphertext<DCRTPoly>& cipher, int d) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        auto vec = ptx->GetRealPackedValue();
        double tr = 0.0;
        for (int i = 0; i < d; i++) {
            tr += vec[i * d + i];
        }
        return tr;
    }

public:
    bool m_verbose = false;

    LinearRegressionBase(std::shared_ptr<Encryption> enc,
                        CryptoContext<DCRTPoly> cc,
                        KeyPair<DCRTPoly> keyPair,
                        std::vector<int> rotIndices)
        : m_enc(enc)
        , m_cc(cc)
        , m_keyPair(keyPair)
        , m_rotIndices(rotIndices)
        , rot(cc, rotIndices, cc->GetRingDimension() / 2) {}

    void setVerbose(bool v) { m_verbose = v; }

    const Ciphertext<DCRTPoly>& getWeights() const { return m_weights; }

    virtual ~LinearRegressionBase() = default;

    virtual TimingResult trainWithTimings(const Ciphertext<DCRTPoly>& X,
                                        const Ciphertext<DCRTPoly>& y) = 0;

    double inferenceAndCalculateMSE(const std::string& testFile, const std::string& saveFile) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, m_weights, &ptx);
        ptx->SetLength(FEATURE_DIM*FEATURE_DIM);
        std::vector<double> weight_vec = ptx->GetRealPackedValue();

        std::vector<double> test_features;
        std::vector<double> test_outcomes;
        CSVProcessor::processDataset(testFile, test_features, test_outcomes,
                                   FEATURE_DIM, SAMPLE_DIM);

        return utils::calculateMSE(weight_vec, test_features, test_outcomes, saveFile, FEATURE_DIM, SAMPLE_DIM);
    }
};
