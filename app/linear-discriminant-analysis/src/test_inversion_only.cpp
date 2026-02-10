// test_inversion_only.cpp
// Quick test of matrix inversion using pre-computed S_W from plaintext results
// This allows fast debugging of the inversion algorithm without running full LDA

#include "lda_data_encoder.h"
#include "lda_newcol.h"
#include "encryption.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace lbcrypto;

// Pre-computed S_W from plaintext run (Heart Disease, 64 samples, 13 features)
// This is the correct S_W that should produce 81.97% accuracy
std::vector<double> getPrecomputedSw() {
    // From plaintext_hd_results_n64.txt
    // Row-major order, 13x13 actual features (padded to 16x16)
    std::vector<double> Sw_13x13 = {
        11.2115, -1.9872,  1.8718,  0.3462,  2.1442, -1.0064, -0.9679, -0.3162,  2.9679, -2.4215, -1.1859,  2.5512,  1.4487,
        -1.9872, 51.9423, -1.7308,  1.0385, -6.9263,  9.9038,  0.1154, -1.7302,  9.4808, -1.1310, -2.8077, -3.9936,  8.0513,
         1.8718, -1.7308, 16.5256, -0.9744,  2.1314, -3.3333,  0.0897, -2.7427,  0.5000, -1.9017, -1.7308, -0.2692, -2.8590,
         0.3462,  1.0385, -0.9744,  9.8590, -0.7147,  0.0897,  5.3333,  0.5918,  3.7821, -0.9786, -0.6282, -2.5897, -0.7308,
         2.1442, -6.9263,  2.1314, -0.7147,  5.6826,  0.2692,  0.2885, -0.0449,  1.9679, -0.7091,  0.9231,  1.9743,  0.6346,
        -1.0064,  9.9038, -3.3333,  0.0897,  0.2692, 23.2308, -4.0000,  0.2885,  2.9231,  1.6449, -3.2692, -4.3462,  2.2692,
        -0.9679,  0.1154,  0.0897,  5.3333,  0.2885, -4.0000, 60.6410, -3.9974,  1.2692,  0.4594,  7.0385,  7.9231, -6.7308,
        -0.3162, -1.7302, -2.7427,  0.5918, -0.0449,  0.2885, -3.9974,  8.9744, -4.3462, -2.9274, -4.3462, -1.1026,  0.0192,
         2.9679,  9.4808,  0.5000,  3.7821,  1.9679,  2.9231,  1.2692, -4.3462, 37.0385, -0.3145,  2.2692, -1.2308,  7.9231,
        -2.4215, -1.1310, -1.9017, -0.9786, -0.7091,  1.6449,  0.4594, -2.9274, -0.3145,  9.7043,  6.5470, -1.4060,  4.7381,
        -1.1859, -2.8077, -1.7308, -0.6282,  0.9231, -3.2692,  7.0385, -4.3462,  2.2692,  6.5470, 19.6923, -1.1538,  3.0769,
         2.5512, -3.9936, -0.2692, -2.5897,  1.9743, -4.3462,  7.9231, -1.1026, -1.2308, -1.4060, -1.1538, 10.2051,  0.5769,
         1.4487,  8.0513, -2.8590, -0.7308,  0.6346,  2.2692, -6.7308,  0.0192,  7.9231,  4.7381,  3.0769,  0.5769, 36.5769
    };

    // Pad to 16x16
    std::vector<double> Sw_16x16(16 * 16, 0.0);
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 13; j++) {
            Sw_16x16[i * 16 + j] = Sw_13x13[i * 13 + j];
        }
    }

    return Sw_16x16;
}

// Pre-computed class means from plaintext run
std::vector<double> getMu0() {
    return {-0.2949, 0.0556, 0.1852, -0.3140, -0.5594, -0.8077, 0.1795, 0.2972, -0.6154, -0.6897, -0.3590, -0.7179, -0.5385};
}

std::vector<double> getMu1() {
    return {0.1071, 0.8571, 0.9286, -0.0744, -0.4800, -0.7143, 0.3929, 0.1276, 0.5714, -0.2551, -0.0714, -0.0238, 0.6429};
}

// Expected S_W^{-1} from plaintext (for verification)
std::vector<double> getExpectedSwInv() {
    std::vector<double> SwInv_13x13 = {
        0.112929, -0.000188, -0.013619, -0.008972, -0.033792, -0.001050, -0.003604, -0.005547, -0.010102, 0.019199, 0.000618, -0.023000, -0.001251,
        -0.000188, 0.025310, 0.001346, -0.000447, 0.033141, -0.009668, -0.003653, 0.001660, -0.005102, -0.000168, 0.002016, 0.006665, -0.003989,
        -0.013619, 0.001346, 0.074010, 0.010308, -0.020393, 0.006684, -0.002809, 0.016626, -0.001091, 0.010089, 0.005133, -0.001131, 0.005839,
        -0.008972, -0.000447, 0.010308, 0.121108, 0.011189, -0.005961, -0.006981, -0.001556, -0.011694, 0.016430, 0.002135, 0.029189, 0.000424,
        -0.033792, 0.033141, -0.020393, 0.011189, 0.261814, -0.010206, -0.004296, 0.003389, -0.006695, 0.021047, -0.003697, -0.036124, -0.006252,
        -0.001050, -0.009668, 0.006684, -0.005961, -0.010206, 0.055509, 0.005106, 0.003247, -0.003192, -0.008877, 0.011066, 0.022419, -0.002405,
        -0.003604, -0.003653, -0.002809, -0.006981, -0.004296, 0.005106, 0.023620, 0.012180, 0.000210, -0.004135, -0.007437, -0.018207, 0.004879,
        -0.005547, 0.001660, 0.016626, -0.001556, 0.003389, 0.003247, 0.012180, 0.138155, 0.013155, 0.033148, 0.021817, 0.009232, 0.000478,
        -0.010102, -0.005102, -0.001091, -0.011694, -0.006695, -0.003192, 0.000210, 0.013155, 0.036907, 0.000063, -0.002620, 0.003631, -0.006093,
        0.019199, -0.000168, 0.010089, 0.016430, 0.021047, -0.008877, -0.004135, 0.033148, 0.000063, 0.159411, 0.051001, 0.015858, -0.017660,
        0.000618, 0.002016, 0.005133, 0.002135, -0.003697, 0.011066, -0.007437, 0.021817, -0.002620, 0.051001, 0.083587, 0.005991, -0.004618,
        -0.023000, 0.006665, -0.001131, 0.029189, -0.036124, 0.022419, -0.018207, 0.009232, 0.003631, 0.015858, 0.005991, 0.133251, -0.003920,
        -0.001251, -0.003989, 0.005839, 0.000424, -0.006252, -0.002405, 0.004879, 0.000478, -0.006093, -0.017660, -0.004618, -0.003920, 0.035018
    };

    std::vector<double> SwInv_16x16(16 * 16, 0.0);
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 13; j++) {
            SwInv_16x16[i * 16 + j] = SwInv_13x13[i * 13 + j];
        }
    }
    return SwInv_16x16;
}

void printMatrix(const std::string& name, const std::vector<double>& M, int rows, int cols, int stride) {
    std::cout << "=== " << name << " (" << rows << "x" << cols << ") ===" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << M[i * stride + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<int> generateRotationIndices(int maxDim) {
    std::vector<int> rotations;
    int batchSize = maxDim * maxDim;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    return rotations;
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Quick Inversion Test (Using Pre-computed S_W)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int f = 13;      // actual features
    int f_tilde = 16; // padded features

    // Get pre-computed values
    auto Sw = getPrecomputedSw();
    auto mu0 = getMu0();
    auto mu1 = getMu1();
    auto expectedSwInv = getExpectedSwInv();

    std::cout << "--- Pre-computed S_W ---" << std::endl;
    printMatrix("S_W", Sw, f, f, f_tilde);

    // Compute trace for reference
    double trace = 0.0;
    for (int i = 0; i < f; i++) {
        trace += Sw[i * f_tilde + i];
    }
    std::cout << "Trace(S_W) = " << trace << std::endl;

    // ========== Setup CKKS Encryption ==========
    std::cout << "\n--- Setting up CKKS Encryption ---" << std::endl;

    int maxDim = f_tilde;
    int multDepth = 29;
    uint32_t scalingModSize = 59;
    uint32_t firstModSize = 60;

    auto rotIndices = generateRotationIndices(maxDim);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scalingModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetBatchSize(maxDim * maxDim);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);

    std::cout << "Setting up bootstrapping..." << std::flush;
    std::vector<uint32_t> levelBudget = {4, 5};
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, f_tilde * f_tilde);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, f_tilde * f_tilde);
    std::cout << " Done." << std::endl;

    // Create encryption object
    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);

    // ========== Encrypt S_W ==========
    std::cout << "\n--- Encrypting S_W ---" << std::endl;
    auto SwPtx = cc->MakeCKKSPackedPlaintext(Sw, 1, 0, nullptr, f_tilde * f_tilde);
    auto SwEnc = cc->Encrypt(SwPtx, keyPair.publicKey);
    std::cout << "S_W encrypted. Level: " << SwEnc->GetLevel() << std::endl;

    // ========== Test Inversion ==========
    std::cout << "\n--- Testing Matrix Inversion (NewCol) ---" << std::endl;

    // Create LDA_NewCol instance
    LDA_NewCol lda(enc, cc, keyPair, rotIndices, multDepth, true);
    lda.m_verbose = true;

    auto start = std::chrono::high_resolution_clock::now();

    // Use new power series algorithm
    // s=64 (samples), f=13 (features), f_tilde=16 (padded)
    int s = 64;  // Number of samples
    int scalarIters = 3;
    int matrixIters = 25;  // More iterations for testing
    auto SwInvEnc = lda.eval_inverse_with_params(SwEnc, f, f_tilde, s, scalarIters, matrixIters);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "\nInversion time: " << duration.count() << " seconds" << std::endl;

    // Decrypt result
    Plaintext ptx;
    cc->Decrypt(keyPair.secretKey, SwInvEnc, &ptx);
    std::vector<double> SwInvResult = ptx->GetRealPackedValue();
    SwInvResult.resize(f_tilde * f_tilde);

    std::cout << "\n--- Results ---" << std::endl;
    printMatrix("Computed S_W^{-1}", SwInvResult, f, f, f_tilde);
    printMatrix("Expected S_W^{-1}", expectedSwInv, f, f, f_tilde);

    // Compute error
    double maxError = 0.0, totalError = 0.0;
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            double err = std::abs(SwInvResult[i * f_tilde + j] - expectedSwInv[i * f_tilde + j]);
            maxError = std::max(maxError, err);
            totalError += err;
        }
    }
    std::cout << "Max Error: " << maxError << std::endl;
    std::cout << "Total Error: " << totalError << std::endl;
    std::cout << "Avg Error: " << totalError / (f * f) << std::endl;

    // ========== Compute Eigenvector ==========
    std::cout << "\n--- Computing Eigenvector ---" << std::endl;

    // w = S_W^{-1} * (mu_1 - mu_0)
    std::vector<double> mu_diff(f_tilde, 0.0);
    for (int i = 0; i < f; i++) {
        mu_diff[i] = mu1[i] - mu0[i];
    }

    std::cout << "mu_1 - mu_0: ";
    for (int i = 0; i < f; i++) {
        std::cout << std::setprecision(4) << mu_diff[i] << " ";
    }
    std::cout << std::endl;

    // Matrix-vector multiply (plaintext for verification)
    std::vector<double> w(f, 0.0);
    for (int i = 0; i < f; i++) {
        for (int j = 0; j < f; j++) {
            w[i] += SwInvResult[i * f_tilde + j] * mu_diff[j];
        }
    }

    std::cout << "Eigenvector w: ";
    for (int i = 0; i < f; i++) {
        std::cout << std::setprecision(6) << w[i] << " ";
    }
    std::cout << std::endl;

    // Expected eigenvector from plaintext
    std::cout << "\nExpected (from plaintext): ";
    std::vector<double> w_expected = {-0.002293, 0.002355, 0.005439, 0.002632, -0.008568, 0.002988, 0.000597, 0.002016, 0.004024, 0.005929, 0.002185, 0.010316, 0.005160};
    for (int i = 0; i < f; i++) {
        std::cout << std::setprecision(6) << w_expected[i] << " ";
    }
    std::cout << std::endl;

    // ========== Compute Projected Means ==========
    double proj_mu0 = 0.0, proj_mu1 = 0.0;
    for (int i = 0; i < f; i++) {
        proj_mu0 += w[i] * mu0[i];
        proj_mu1 += w[i] * mu1[i];
    }

    std::cout << "\nProjected means:" << std::endl;
    std::cout << "  proj_mu_0 = " << proj_mu0 << " (expected: -0.004917)" << std::endl;
    std::cout << "  proj_mu_1 = " << proj_mu1 << " (expected: 0.018929)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Complete!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
