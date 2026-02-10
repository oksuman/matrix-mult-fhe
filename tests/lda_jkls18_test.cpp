// Simple test for LDA's JKLS18 implementation (FIXED version)
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "encryption.h"
#include "rotation.h"
#include "openfhe.h"

using namespace lbcrypto;

// Fixed JKLS18 implementation matching matrix_algo_singlePack.h
class LDA_JKLS18_Test {
private:
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    RotationComposer rot;

    std::vector<double> vectorRotate(const std::vector<double>& vec, int rotation) {
        int n = vec.size();
        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = vec[((i + rotation) % n + n) % n];
        }
        return result;
    }

    Ciphertext<DCRTPoly> getZeroCiphertext(int batchSize) {
        std::vector<double> zeroVec(batchSize, 0.0);
        auto zeroPtx = m_cc->MakeCKKSPackedPlaintext(zeroVec, 1, 0, nullptr, batchSize);
        return m_cc->Encrypt(zeroPtx, m_keyPair.publicKey);
    }

    // Generate sigma mask for JKLS18 (diagonal at offset k)
    std::vector<double> generateSigmaMsk_JKLS18(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d; i++) {
            msk[(i * d) + ((i + k) % d)] = 1;
        }
        return msk;
    }

    // Generate tau mask for JKLS18
    std::vector<double> generateTauMsk_JKLS18(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d; i++) {
            msk[(((i + k) % d) * d) + i] = 1;
        }
        return msk;
    }

    // Generate shifting mask for JKLS18
    std::vector<double> generateShiftingMsk_JKLS18(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d; i++) {
            msk[(i * d) + ((d - k + i) % d)] = 1;
        }
        return msk;
    }

    // Column shifting operation
    Ciphertext<DCRTPoly> columnShifting_JKLS18(Ciphertext<DCRTPoly> M, int k, int d) {
        auto msk = generateShiftingMsk_JKLS18(k, d);
        auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
        auto result = m_cc->EvalMult(M, pmsk);
        return rot.rotate(result, k);
    }

    // Sigma transform with baby-step giant-step optimization
    Ciphertext<DCRTPoly> sigmaTransform_JKLS18(Ciphertext<DCRTPoly> M, int d) {
        int steps = static_cast<int>(std::sqrt(d));

        // Generate baby steps
        std::vector<Ciphertext<DCRTPoly>> babySteps(steps);
        for (int i = 0; i < steps; i++) {
            babySteps[i] = rot.rotate(M, -i);
        }

        // sigma_M accumulator
        auto sigma_M = getZeroCiphertext(d * d);

        // First group (j = steps-1): doesn't need giant step rotation
        {
            for (int i = 0; i < steps; i++) {
                auto msk = generateSigmaMsk_JKLS18(steps * (steps - 1) + i, d);
                msk = vectorRotate(msk, i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(sigma_M,
                    m_cc->EvalMult(babySteps[i], pmsk));
            }
        }

        // Remaining groups
        for (int j = 0; j < steps - 1; j++) {
            auto tmp = getZeroCiphertext(d * d);

            for (int i = 0; i < steps; i++) {
                auto msk = generateSigmaMsk_JKLS18(steps * j + i, d);
                msk = vectorRotate(msk, i + steps * j);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[i], pmsk));
            }
            m_cc->EvalAddInPlace(sigma_M, rot.rotate(tmp, -steps * j));
        }

        return sigma_M;
    }

    // Tau transform with baby-step giant-step optimization
    Ciphertext<DCRTPoly> tauTransform_JKLS18(Ciphertext<DCRTPoly> M, int d) {
        int steps = static_cast<int>(std::sqrt(d));

        // Generate baby steps with stride d
        std::vector<Ciphertext<DCRTPoly>> babySteps(steps);
        for (int i = 0; i < steps; i++) {
            babySteps[i] = rot.rotate(M, -i * d);
        }

        // tau_M accumulator
        auto tau_M = getZeroCiphertext(d * d);

        // First group (i = steps-1): doesn't need giant step rotation
        {
            for (int j = 0; j < steps; j++) {
                auto msk = generateTauMsk_JKLS18(steps * (steps - 1) + j, d);
                msk = vectorRotate(msk, j * d);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tau_M,
                    m_cc->EvalMult(rot.rotate(M, (steps * (steps - 1) + j) * d), pmsk));
            }
        }

        // Remaining groups
        for (int i = 0; i < steps - 1; i++) {
            auto tmp = getZeroCiphertext(d * d);

            for (int j = 0; j < steps; j++) {
                auto msk = generateTauMsk_JKLS18(steps * i + j, d);
                msk = vectorRotate(msk, -steps * d * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
            }
            m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, steps * d * i));
        }

        return tau_M;
    }

public:
    LDA_JKLS18_Test(CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keyPair, std::vector<int> rotIndices)
        : m_cc(cc), m_keyPair(keyPair), rot(cc, rotIndices, cc->GetRingDimension() / 2) {}

    Ciphertext<DCRTPoly> eval_mult_JKLS18(const Ciphertext<DCRTPoly>& A,
                                          const Ciphertext<DCRTPoly>& B, int d) {
        auto sigma_A = sigmaTransform_JKLS18(A, d);
        auto tau_B = tauTransform_JKLS18(B, d);
        auto matrixC = m_cc->EvalMultAndRelinearize(sigma_A, tau_B);

        for (int i = 1; i < d; i++) {
            auto shifted_A = columnShifting_JKLS18(sigma_A, i, d);
            tau_B = rot.rotate(tau_B, d);
            m_cc->EvalAddInPlace(
                matrixC, m_cc->EvalMultAndRelinearize(shifted_A, tau_B));
        }

        return matrixC;
    }
};

void printMatrix(const std::vector<double>& mat, int d, const std::string& name) {
    std::cout << name << " (" << d << "x" << d << "):" << std::endl;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << mat[i * d + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<double> matmul_plain(const std::vector<double>& A, const std::vector<double>& B, int d) {
    std::vector<double> C(d * d, 0.0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                C[i * d + j] += A[i * d + k] * B[k * d + j];
            }
        }
    }
    return C;
}

int main() {
    const int d = 4;  // Small test first

    std::cout << "Testing LDA's JKLS18 implementation (FIXED) with " << d << "x" << d << " matrices\n\n";

    // Setup CKKS
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(5);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d * d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    // Create test matrices
    std::vector<double> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    std::vector<double> B = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };  // Identity matrix

    printMatrix(A, d, "A");
    printMatrix(B, d, "B (Identity)");

    // Plaintext multiplication
    auto C_plain = matmul_plain(A, B, d);
    printMatrix(C_plain, d, "Expected A*B (plaintext)");

    // Encrypted multiplication using LDA's JKLS18
    auto enc = std::make_shared<DebugEncryption>(cc, keyPair);
    LDA_JKLS18_Test jkls18(cc, keyPair, rotations);

    auto enc_A = enc->encryptInput(A);
    auto enc_B = enc->encryptInput(B);

    auto enc_C = jkls18.eval_mult_JKLS18(enc_A, enc_B, d);

    Plaintext result;
    cc->Decrypt(keyPair.secretKey, enc_C, &result);
    auto C_dec = result->GetRealPackedValue();
    C_dec.resize(d * d);

    printMatrix(C_dec, d, "Encrypted A*B (LDA JKLS18)");

    // Check error
    double maxError = 0.0;
    for (int i = 0; i < d * d; i++) {
        maxError = std::max(maxError, std::abs(C_plain[i] - C_dec[i]));
    }
    std::cout << "Max error: " << maxError << std::endl;

    if (maxError < 0.01) {
        std::cout << "\nTEST PASSED!\n";
    } else {
        std::cout << "\nTEST FAILED!\n";
    }

    return 0;
}
