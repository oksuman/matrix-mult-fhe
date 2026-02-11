// test_inversion_only.cpp
// Quick test of matrix inversion using pre-computed S_W from plaintext results
// This allows fast debugging of the inversion algorithm without running full LDA

#include "lda_data_encoder.h"
#include "lda_ar24.h"
#include "encryption.h"
#include <openfhe.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace lbcrypto;

// S_W from actual LDA run (Heart Disease, 64 samples, 13 features, trace=691.9)
// This is the S_W that produced zeros in AR24 inversion
std::vector<double> getPrecomputedSw() {
    // From actual encrypted LDA run with 64 samples
    // Row-major order, 13x13 actual features (padded to 16x16)
    std::vector<double> Sw_13x13 = {
        58.6974,  -3.5278,  14.1253,  20.8481,  22.5481,   7.5208,  17.6797, -23.4590,   7.5180,   8.9947,   6.1894,  19.0222,   9.9991,
        -3.5278,  54.2291,  -6.2869,  -0.1273, -13.9894,  17.8023,  -4.5308,  -3.0022,  10.6812,  -7.3789,  -7.7351,  -9.6560,   8.4133,
        14.1253,  -6.2869,  49.7587,  -1.5966,   3.3445, -10.4177,   2.3190, -18.5653,   9.8208,  -1.6076,  -1.9796,   8.5707,  -0.4942,
        20.8481,  -0.1273,  -1.5966,  63.5466,   7.9160,  23.3991,  13.5337,  -2.7706,   0.8076,   6.3440,   0.7574,  -1.1563,  -0.3090,
        22.5481, -13.9894,   3.3445,   7.9160,  63.5534,  -2.4244,  17.2481,  -3.7799,   4.6728,   1.2747,   2.2540,   7.6726,   4.4115,
         7.5208,  17.8023, -10.4177,  23.3991,  -2.4244,  63.2642,   4.3063,  -1.3702,   6.2768,  -2.3836,  13.0538, -10.3543,   9.2582,
        17.6797,  -4.5308,   2.3190,  13.5337,  17.2481,   4.3063,  61.2215,  -2.8400,   0.1735,   2.4889,   8.3781,  11.4494,  -5.6838,
       -23.4590,  -3.0022, -18.5653,  -2.7706,  -3.7799,  -1.3702,  -2.8400,  48.9188, -17.7847, -13.6963, -17.7071,  -4.0702,  -3.9345,
         7.5180,  10.6812,   9.8208,   0.8076,   4.6728,   6.2768,   0.1735, -17.7847,  44.4275,  -1.1515,   2.3564,  -3.4913,   7.1571,
         8.9947,  -7.3789,  -1.6076,   6.3440,   1.2747,  -2.3836,   2.4889, -13.6963,  -1.1515,  46.4802,  33.5021,   2.8815,  10.5891,
         6.1894,  -7.7351,  -1.9796,   0.7574,   2.2540,  13.0538,   8.3781, -17.7071,   2.3564,  33.5021,  53.5425,  -0.5553,   1.0502,
        19.0222,  -9.6560,   8.5707,  -1.1563,   7.6726, -10.3543,  11.4494,  -4.0702,  -3.4913,   2.8815,  -0.5553,  38.0112,  -4.7923,
         9.9991,   8.4133,  -0.4942,  -0.3090,   4.4115,   9.2582,  -5.6838,  -3.9345,   7.1571,  10.5891,   1.0502,  -4.7923,  46.2627
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

    int maxDim = 64;  // AR24 uses d*d*s = 16*16*16 = 4096 slots, need rotation keys for 64*64
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
    std::cout << "\n--- Testing Matrix Inversion (AR24) ---" << std::endl;

    // Create LDA_AR24 instance
    LDA_AR24 lda(enc, cc, keyPair, rotIndices, multDepth, true);
    lda.m_verbose = true;

    auto start = std::chrono::high_resolution_clock::now();

    // Full 25 iterations to test bootstrapping properly
    int matrixIters = 25;
    double traceUpperBound = 64.0 * f;  // samples * features
    auto SwInvEnc = lda.eval_inverse_impl(SwEnc, f_tilde, matrixIters, f, traceUpperBound);

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
