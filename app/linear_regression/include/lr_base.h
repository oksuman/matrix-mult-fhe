// app/linear_regression/include/lr_base.h
#pragma once

#include "rotation.h"
#include "matrix_algo_singlePack.h"
#include "matrix_inversion_algo.h"
#include "csv_processor.h"
#include "mse_calculator.h"
#include <memory>
#include <chrono>
#include <tuple>

constexpr int FEATURE_DIM = 8;
constexpr int SAMPLE_DIM = 64;

class LinearRegressionBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    std::vector<int> m_rotIndices;
    Ciphertext<DCRTPoly> m_weights;
    RotationComposer rot;

    using TimingResult = std::tuple<std::chrono::duration<double>,
                                   std::chrono::duration<double>,
                                   std::chrono::duration<double>,
                                   std::chrono::duration<double>>;

    // Common utility for all algorithms
    Ciphertext<DCRTPoly> computeXty(const Ciphertext<DCRTPoly>& Xt,
                                   const Ciphertext<DCRTPoly>& y) {
        auto result = y->Clone();
        // Replicate 
        for(int i = 0; i < log2(FEATURE_DIM); i++) {
            m_cc->EvalAddInPlace(result, 
                rot.rotate(result, -SAMPLE_DIM * (1 << i)));
        }    
        // Multiply with X^t
        result = m_cc->EvalMultAndRelinearize(Xt, result);
        return result;
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

    double inferenceAndCalculateMSE(const std::string& testFile) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, m_weights, &ptx);
        std::vector<double> weight_vec = ptx->GetRealPackedValue();

        std::vector<double> test_features;
        std::vector<double> test_outcomes;
        CSVProcessor::processDataset(testFile, test_features, test_outcomes, 
                                   FEATURE_DIM, SAMPLE_DIM);

        return utils::calculateMSE(weight_vec, test_features, test_outcomes);
    }
};