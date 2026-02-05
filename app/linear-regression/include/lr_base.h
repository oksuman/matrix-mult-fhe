// lr_base.h
#pragma once

#include "encryption.h"
#include "rotation.h"
#include "csv_processor.h"
#include "mse_calculator.h"
#include <memory>
#include <chrono>
#include <tuple>

const int FEATURE_DIM = 8;
const int SAMPLE_DIM = 64;


class LinearRegressionBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    std::vector<int> m_rotIndices;
    Ciphertext<DCRTPoly> m_weights;
    RotationComposer rot;

    using TimingResult = std::tuple<std::chrono::duration<double>,    // Step 1: XtX time
                                   std::chrono::duration<double>,      // Step 2: Matrix inverse time
                                   std::chrono::duration<double>,      // Step 3: Xty time
                                   std::chrono::duration<double>>;     // Step 4: Final weights time

    // Common utility functions
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

    // Matrix trace operation
    Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        std::vector<double> msk(batchSize, 0);
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }
        auto trace = m_cc->EvalMult(M, m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, batchSize));

        for (int i = 1; i <= log2(batchSize); i++) {
            m_cc->EvalAddInPlace(trace,
                                 rot.rotate(trace, batchSize / (1 << i)));
        }
        return trace;
    }

    // Matrix transpose operation
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

    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        auto p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(0, d), 1, 0, nullptr, batchSize);
        auto M_transposed = m_cc->EvalMult(M, p);

        for (int i = 1; i < d; i++) {
            p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(i, d), 1, 0, nullptr, batchSize);
            m_cc->EvalAddInPlace(M_transposed,
                                m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }

        for (int i = -1; i > -d; i--) {
            p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(i, d), 1, 0, nullptr, batchSize);
            m_cc->EvalAddInPlace(M_transposed,
                                m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }
        return M_transposed;
    }

    // X^t y computation
    Ciphertext<DCRTPoly> computeXty(const Ciphertext<DCRTPoly>& Xt,
                                   const Ciphertext<DCRTPoly>& y, 
                                   int featureDim, int sampleDim) {
        auto result = y->Clone();
        // Replicate 
        for(int i = 0; i < log2(featureDim); i++) {
            m_cc->EvalAddInPlace(result, 
                rot.rotate(result, -sampleDim * (1 << i)));
        }    
    
        // Multiply with X^t
        result = m_cc->EvalMultAndRelinearize(Xt, result);
        
        result = eval_transpose(result, SAMPLE_DIM, SAMPLE_DIM * SAMPLE_DIM);

        // Sum rows of result
        for(int i = 0; i < log2(SAMPLE_DIM); i++) {
            m_cc->EvalAddInPlace(result, 
                rot.rotate(result, -SAMPLE_DIM * (1 << i)));
        }
        // Sum to feature size
        for(int i = 0; i < log2(SAMPLE_DIM / FEATURE_DIM); i++) {
            m_cc->EvalAddInPlace(result, 
                rot.rotate(result, -FEATURE_DIM * (1 << i)));
        }
        result->SetSlots(FEATURE_DIM * FEATURE_DIM);

        return result;
    }

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

public:
    LinearRegressionBase(std::shared_ptr<Encryption> enc,
                        CryptoContext<DCRTPoly> cc,
                        KeyPair<DCRTPoly> keyPair,
                        std::vector<int> rotIndices)
        : m_enc(enc)
        , m_cc(cc)
        , m_keyPair(keyPair)
        , m_rotIndices(rotIndices)
        , rot(cc, rotIndices, cc->GetRingDimension() / 2) {}

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