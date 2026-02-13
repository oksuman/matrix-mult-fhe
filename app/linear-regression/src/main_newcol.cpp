// app/linear_regression/src/main.cpp
#include "lr_newcol.h"
#include <iostream>
#include <fstream>
#include "encryption.h"
#include "rotation.h"

// const int FEATURE_DIM = 8;
// const int SAMPLE_DIM = 64;

int main() {
    int multDepth = 29; 
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);

    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);

    parameters.SetBatchSize(SAMPLE_DIM * SAMPLE_DIM);
    parameters.SetSecurityLevel(HEStd_128_classic);

    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};


    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
        

    auto keyPair = cc->KeyGen();

    // Setup rotation keys
    std::vector<int> rotations;
    for (int i = 1; i < 1<<16; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, FEATURE_DIM * FEATURE_DIM);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, FEATURE_DIM * FEATURE_DIM);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

    // Create LinearRegression instance
    LinearRegression_NewCol lr(enc, cc, keyPair, rotations, multDepth);
    lr.setVerbose(true);  // Enable debug output

    // Process training data
    std::vector<double> features;
    std::vector<double> outcomes;
    CSVProcessor::processDataset(std::string(DATA_DIR) + "/trainSet.csv", features, outcomes,
                               FEATURE_DIM, SAMPLE_DIM);
    
    // Encrypt data
    // X: 64x64 matrix (SAMPLE_DIM*SAMPLE_DIM = 4096 slots)
    // y: 64 elements (SAMPLE_DIM slots) - will be replicated via SetSlots in computeXty
    auto X = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(features, 1, 0, nullptr, SAMPLE_DIM * SAMPLE_DIM));
    auto y = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(outcomes, 1, 0, nullptr, SAMPLE_DIM));

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
    double mse = lr.inferenceAndCalculateMSE(std::string(DATA_DIR) + "/testSet.csv", "newcol_mse_result.txt");
    std::cout << "mse: " << mse << std::endl;
    return 0;
}