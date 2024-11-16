#pragma once

#include "lr_base.h"
#include "matrix_algo_singlePack.h"
#include "matrix_inversion_algo.h"
#include "utils/csv_processor.h"
#include "utils/mse_calculator.h"
#include <memory>
#include <chrono>
#include <vector>
#include <string>


class LinearRegression_JKLS18 : public LinearRegressionBase<
    MatrixMult_JKLS18<SAMPLE_DIM>,
    MatrixInverse_JKLS18<FEATURE_DIM>> {
private:
    using Base = LinearRegressionBase<
        MatrixMult_JKLS18<SAMPLE_DIM>,
        MatrixInverse_JKLS18<FEATURE_DIM>>;

public:
    LinearRegression_JKLS18(std::shared_ptr<Encryption> enc,
                           CryptoContext<DCRTPoly> cc,
                           KeyPair<DCRTPoly> keyPair,
                           std::vector<int> rotIndices)
        : Base(enc, cc, keyPair,
               std::make_unique<MatrixMult_JKLS18<SAMPLE_DIM>>(
                   enc, cc, keyPair.publicKey, rotIndices),
               std::make_unique<MatrixInverse_JKLS18<FEATURE_DIM>>(
                   enc, cc, keyPair.publicKey, rotIndices, 22, 79)) {} //r, depth

    typename Base::TimingResult 
    trainWithTimings(const Ciphertext<DCRTPoly>& X,
                    const Ciphertext<DCRTPoly>& y) override {
        using namespace std::chrono;
        
        // Step 1: X^tX
        auto step1_start = high_resolution_clock::now();
        auto Xt = m_matMult->eval_transpose(X);
        auto XtX = m_matMult->eval_mult(Xt, X);
        // re-batch
        auto rebatched_XtX = XtX; 
        for(int i=0; i<FEATURE_DIM-1; i++){
            this->m_cc->EvalAddInPlace(rebatched_XtX, rot.rotate(XtX, (SAMPLE_DIM - FEATURE_DIM)*(i+1)));
        }
        std::vector<double> msk(FEATURE_DIM*FEATURE_DIM, 1.0);
        rebatched_XtX = this->m_cc->EvalMult(rebatched_XtX, this->m_cc->MakeCKKSPackedPlaintext(msk));

        auto step1_end = high_resolution_clock::now();

        // Step 2: Matrix inverse
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = m_matInv->eval_inverse(XtX);
        auto step2_end = high_resolution_clock::now();

        // Step 3: X^ty
        auto step3_start = high_resolution_clock::now();
        auto Xty = this->computeXty(Xt, y);
        auto step3_end = high_resolution_clock::now();

        // Step 4: Final weight computation
        auto step4_start = high_resolution_clock::now();
        this->m_weights = m_matInv->eval_mult(inv_XtX, Xty);
        auto step4_end = high_resolution_clock::now();

        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }
};