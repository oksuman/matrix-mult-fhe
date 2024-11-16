#pragma once

#include "lr_base.h"
#include "matrix_algo_singlePack.h"
#include "matrix_inversion_algo.h"
#include "csv_processor.h"
#include "mse_calculator.h"
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include "rotation.h"

class LinearRegression_NewCol : public LinearRegressionBase {
private:
    std::unique_ptr<MatrixMult_newCol<SAMPLE_DIM>> m_matMult;
    std::unique_ptr<MatrixInverse_newCol<FEATURE_DIM>> m_matInv;

public:
    LinearRegression_NewCol(std::shared_ptr<Encryption> enc,
                          CryptoContext<DCRTPoly> cc,
                          KeyPair<DCRTPoly> keyPair,
                          std::vector<int> rotIndices)
        : LinearRegressionBase(enc, cc, keyPair, rotIndices)
        , m_matMult(std::make_unique<MatrixMult_newCol<SAMPLE_DIM>>(
              enc, cc, keyPair.publicKey, rotIndices))
        , m_matInv(std::make_unique<MatrixInverse_newCol<FEATURE_DIM>>(
              enc, cc, keyPair.publicKey, rotIndices, 18, 48)) {} // lr-48, inverse-48

    TimingResult trainWithTimings(const Ciphertext<DCRTPoly>& X,
                                 const Ciphertext<DCRTPoly>& y) override {
        using namespace std::chrono;
        
        // Step 1: X^tX
        auto step1_start = high_resolution_clock::now();
        auto Xt = m_matMult->eval_transpose(X);
        auto XtX = m_matMult->eval_mult(Xt, X);
        std::cout << "XtX level: " << XtX->GetLevel() << std::endl;

        // re-batch
        auto rebatched_XtX = XtX; 
        for(int i=0; i<FEATURE_DIM-1; i++){
            this->m_cc->EvalAddInPlace(rebatched_XtX, rot.rotate(XtX, (SAMPLE_DIM - FEATURE_DIM)*(i+1)));
        }
        std::vector<double> msk(FEATURE_DIM*FEATURE_DIM, 1.0);
        rebatched_XtX = this->m_cc->EvalMult(rebatched_XtX, this->m_cc->MakeCKKSPackedPlaintext(msk));

        for(int i=0; i<log2((SAMPLE_DIM*SAMPLE_DIM)/(FEATURE_DIM*FEATURE_DIM)); i++){
            this->m_cc->EvalAddInPlace(rebatched_XtX, rot.rotate(rebatched_XtX, -SAMPLE_DIM*(1<<i)));
        }
        rebatched_XtX->SetSlots(FEATURE_DIM*FEATURE_DIM);
        auto step1_end = high_resolution_clock::now();

        // Step 2: Matrix inverse
        auto step2_start = high_resolution_clock::now();
        auto inv_XtX = m_matInv->eval_inverse(XtX);
        auto step2_end = high_resolution_clock::now();

        std::cout << "XtX-1 level: " << inv_XtX->GetLevel() << std::endl;

        // Step 3: X^ty
        auto step3_start = high_resolution_clock::now();
        auto Xty = computeXty(Xt, y);
        Xty = m_matInv->eval_transpose(Xty);    
        // Sum rows of result
        for(int i = 0; i < log2(SAMPLE_DIM); i++) {
            this->m_cc->EvalAddInPlace(Xty, 
                rot.rotate(Xty, -SAMPLE_DIM * (1 << i)));
        }
        // Sum to feature size
        for(int i = 0; i < log2(SAMPLE_DIM / FEATURE_DIM); i++) {
            this->m_cc->EvalAddInPlace(Xty, 
                rot.rotate(Xty, -FEATURE_DIM * (1 << i)));
        }
        Xty->SetSlots(FEATURE_DIM * FEATURE_DIM);
        auto step3_end = high_resolution_clock::now();

        // Step 4: Final weight computation
        auto step4_start = high_resolution_clock::now();
        m_weights = m_matInv->eval_mult(inv_XtX, Xty);
        for(int i=0; i<log2(FEATURE_DIM); i++){
            this->m_cc->EvalAddInPlace(m_weights, rot.rotate(m_weights, -1<<i));
        }
        m_weights = m_matInv->eval_transpose(m_weights); 
        m_weights->SetSlots(FEATURE_DIM);
        auto step4_end = high_resolution_clock::now();
        
        std::cout << "final weight level: " << m_weights->GetLevel() << std::endl;
        return {
            step1_end - step1_start,
            step2_end - step2_start,
            step3_end - step3_start,
            step4_end - step4_start
        };
    }
};