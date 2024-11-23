// app/linear_regression/src/main.cpp
#include "lr_newcol.h"
#include <iostream>
#include <fstream>
#include "encryption.h"
#include "rotation.h"

// const int FEATURE_DIM = 8;
// const int SAMPLE_DIM = 64;

int main() {
    int multDepth = 53; 
    uint32_t scaleModSize = 50;
    
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(SAMPLE_DIM * SAMPLE_DIM);
    // parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1<<13);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    // Setup rotation keys
    std::vector<int> rotations;
    for (int i = 1; i < SAMPLE_DIM * SAMPLE_DIM; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

    // Create LinearRegression instance
    LinearRegression_NewCol lr(enc, cc, keyPair, rotations);

    // Process training data
    std::vector<double> features;
    std::vector<double> outcomes;
    CSVProcessor::processDataset("data/trainSet.csv", features, outcomes, 
                               FEATURE_DIM, SAMPLE_DIM);
    
    // Encrypt data
    auto X = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(features));
    auto y = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(outcomes));

    // Open files for results
    std::ofstream timingFile("newcol_timing.txt");
    
    // Train and measure time
    auto [step1_time, step2_time, step3_time, step4_time] = 
        lr.trainWithTimings(X, y);

    // Record timings
    timingFile << "Step 1 (X^tX): " << step1_time.count() << " seconds\n"
               << "Step 2 (Inverse): " << step2_time.count() << " seconds\n"
               << "Step 3 (X^ty): " << step3_time.count() << " seconds\n"
               << "Step 4 (Weights): " << step4_time.count() << " seconds\n"
               << "Total time: " 
               << (step1_time + step2_time + step3_time + step4_time).count() 
               << " seconds\n";
    timingFile.close();

    // Calculate and record MSE
    double mse = lr.inferenceAndCalculateMSE("data/testSet.csv");
    std::cout << "mse: " << mse << std::endl;
    return 0;
}