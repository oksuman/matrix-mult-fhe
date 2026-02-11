// Quick test for AR24 matrix multiplication
#include <iostream>
#include <vector>
#include "openfhe.h"
#include "include/lda_ar24.h"

using namespace lbcrypto;

int main() {
    const int d = 16;

    // Setup CKKS
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetMultiplicativeDepth(15);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d * d * d);  // d*d*s where s=d

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    // Generate rotation indices for AR24
    std::vector<int> rotIndices;
    for (int i = 1; i < d * d * d; i *= 2) {
        rotIndices.push_back(i);
        rotIndices.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotIndices);

    int s = d;  // For AR24 with d=16

    // Create simple test matrices (identity * 2)
    std::vector<double> matA(d * d * s, 0.0);
    std::vector<double> matB(d * d * s, 0.0);

    // A = 2*I (first d*d slots, rest zero)
    for (int i = 0; i < d; i++) {
        matA[i * d + i] = 2.0;
    }

    // B = 3*I (first d*d slots, rest zero)
    for (int i = 0; i < d; i++) {
        matB[i * d + i] = 3.0;
    }

    std::cout << "=== AR24 Matrix Multiplication Test ===" << std::endl;
    std::cout << "d = " << d << ", s = " << s << std::endl;
    std::cout << "A = 2*I, B = 3*I" << std::endl;
    std::cout << "Expected: A*B = 6*I" << std::endl << std::endl;

    // Encrypt
    Plaintext ptA = cc->MakeCKKSPackedPlaintext(matA, 1, 0, nullptr, d * d * s);
    Plaintext ptB = cc->MakeCKKSPackedPlaintext(matB, 1, 0, nullptr, d * d * s);
    auto ctA = cc->Encrypt(keyPair.publicKey, ptA);
    auto ctB = cc->Encrypt(keyPair.publicKey, ptB);

    // Create LDA_AR24 instance for multiplication
    auto enc = std::make_shared<DebugEncryption>(cc, keyPair.secretKey);
    LDA_AR24 lda(enc, cc, keyPair, rotIndices, 15, false);

    // Test multiplication
    std::cout << "Performing AR24 multiplication..." << std::endl;
    auto ctC = lda.eval_mult_AR24_public(ctA, ctB, d, s);

    // Decrypt and check result
    Plaintext ptC;
    cc->Decrypt(keyPair.secretKey, ctC, &ptC);
    auto result = ptC->GetRealPackedValue();

    std::cout << "\nResult C (first " << d << "x" << d << " = " << d*d << " slots):" << std::endl;
    std::cout << "Diagonal elements (should be 6.0):" << std::endl;
    for (int i = 0; i < d; i++) {
        std::cout << "  C[" << i << "," << i << "] = " << result[i * d + i] << std::endl;
    }

    std::cout << "\nOff-diagonal sample (should be ~0):" << std::endl;
    std::cout << "  C[0,1] = " << result[1] << std::endl;
    std::cout << "  C[1,0] = " << result[d] << std::endl;

    std::cout << "\nSlots d*d to d*d+10 (should be ~0, zero padding):" << std::endl;
    for (int i = d*d; i < d*d + 10 && i < (int)result.size(); i++) {
        std::cout << "  slot[" << i << "] = " << result[i] << std::endl;
    }

    return 0;
}
