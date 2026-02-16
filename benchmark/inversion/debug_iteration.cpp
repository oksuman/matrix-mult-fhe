#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace lbcrypto;
using namespace BenchmarkConfig;

constexpr int d = 16;
constexpr int OVERRIDE_ITERATIONS = 0;  // 0 = use default iterations

class DebugIteration {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    std::shared_ptr<Encryption> enc;
    std::vector<double> matrix;
    std::vector<int> rotations;
    std::vector<double> I;  // Identity matrix
    int r;
    int multDepth = 28;

public:
    DebugIteration() : r(OVERRIDE_ITERATIONS > 0 ? OVERRIDE_ITERATIONS : getInversionIterations(d)) {
        uint32_t scaleModSize = 59;
        uint32_t firstModSize = 60;
        std::vector<uint32_t> levelBudget = {4, 4};
        std::vector<uint32_t> bsgsDim = {0, 0};

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetScalingModSize(scaleModSize);
        parameters.SetFirstModSize(firstModSize);
        parameters.SetBatchSize(d * d);
        parameters.SetSecurityLevel(HEStd_128_classic);

        cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);

        keyPair = cc->KeyGen();
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, d * d);
        cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

        for (int i = 1; i < d * d * d; i *= 2) {
            rotations.push_back(i);
            rotations.push_back(-i);
        }
        cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
        cc->EvalMultKeyGen(keyPair.secretKey);

        enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

        // Identity matrix
        I.resize(d * d, 0.0);
        for (int i = 0; i < d; i++) I[i * d + i] = 1.0;

        matrix.resize(d * d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (size_t i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));
    }

    std::vector<double> decrypt(const Ciphertext<DCRTPoly>& ct) {
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, ct, &ptx);
        ptx->SetLength(d * d);
        auto result = ptx->GetRealPackedValue();
        result.resize(d * d);
        return result;
    }

    std::vector<double> ptTranspose(const std::vector<double>& M) {
        std::vector<double> Mt(d * d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                Mt[j * d + i] = M[i * d + j];
        return Mt;
    }

    std::vector<double> ptMult(const std::vector<double>& A, const std::vector<double>& B) {
        std::vector<double> C(d * d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                for (int k = 0; k < d; k++)
                    C[i * d + j] += A[i * d + k] * B[k * d + j];
        return C;
    }

    std::vector<double> ptAdd(const std::vector<double>& A, const std::vector<double>& B) {
        std::vector<double> C(d * d);
        for (int i = 0; i < d * d; i++) C[i] = A[i] + B[i];
        return C;
    }

    std::vector<double> ptScale(const std::vector<double>& A, double s) {
        std::vector<double> C(d * d);
        for (int i = 0; i < d * d; i++) C[i] = A[i] * s;
        return C;
    }

    double ptTrace(const std::vector<double>& M) {
        double tr = 0.0;
        for (int i = 0; i < d; i++) tr += M[i * d + i];
        return tr;
    }

    double maxError(const std::vector<double>& a, const std::vector<double>& b) {
        double err = 0.0;
        for (int i = 0; i < d * d; i++) err = std::max(err, std::abs(a[i] - b[i]));
        return err;
    }

    double frobNorm(const std::vector<double>& a) {
        double sum = 0.0;
        for (int i = 0; i < d * d; i++) sum += a[i] * a[i];
        return std::sqrt(sum);
    }

    void printSample(const std::string& name, const std::vector<double>& v, int n = 4) {
        std::cout << name << " [";
        for (int i = 0; i < n && i < (int)v.size(); i++) {
            std::cout << std::fixed << std::setprecision(6) << v[i];
            if (i < n - 1) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;
    }

    void run() {
        std::cout << "===== Debug Iteration d=" << d << " r=" << r << " =====" << std::endl;
        std::cout << "\nAlgorithm:" << std::endl;
        std::cout << "  A_bar = I - (M * M^T) / trace(M * M^T)" << std::endl;
        std::cout << "  Y_0 = M^T / trace(M * M^T)" << std::endl;
        std::cout << "  Y_{i+1} = Y_i * (I + A_bar_i)" << std::endl;
        std::cout << "  A_bar_{i+1} = A_bar_i * A_bar_i" << std::endl;
        std::cout << std::endl;

        auto matMult = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

        // Step 1: M * M^T (NOT M^T * M)
        auto encM = enc->encryptInput(matrix);
        auto ptMt = ptTranspose(matrix);
        auto ptMMt = ptMult(matrix, ptMt);  // M * M^T

        auto encMt = matMult->eval_transpose(encM);
        auto encMMt = matMult->eval_mult(encM, encMt);  // M * M^T

        std::cout << "M*M^T maxErr: " << std::scientific << maxError(decrypt(encMMt), ptMMt) << std::endl;

        // Step 2: trace and 1/trace
        double ptTr = ptTrace(ptMMt);
        double ptInvTr = 1.0 / ptTr;
        std::cout << "trace(M*M^T) = " << std::fixed << std::setprecision(4) << ptTr << std::endl;
        std::cout << "1/trace = " << std::scientific << ptInvTr << std::endl;

        auto encTrace = matMult->eval_trace(encMMt, d * d);
        auto encInvTr = matMult->eval_scalar_inverse(encTrace, d * d, 4, d * d);

        auto decInvTr = decrypt(encInvTr);
        std::cout << "enc 1/trace = " << decInvTr[0] << " err=" << std::abs(decInvTr[0] - ptInvTr) << std::endl;

        // Step 3: A_bar = I - (M * M^T) / trace
        // Create plaintext with proper slots
        auto pI = cc->MakeCKKSPackedPlaintext(I, 1, 0, nullptr, d * d);
        auto encA_bar = cc->EvalSub(pI, cc->EvalMult(encMMt, encInvTr));
        auto ptA_bar = ptAdd(I, ptScale(ptMMt, -ptInvTr));  // I - MMt/trace

        std::cout << "\nA_bar_0 = I - (M*M^T)/trace:" << std::endl;
        printSample("  pt A_bar", ptA_bar);
        printSample("  enc A_bar", decrypt(encA_bar));
        std::cout << "  maxErr: " << std::scientific << maxError(decrypt(encA_bar), ptA_bar) << std::endl;
        std::cout << "  ||A_bar||_F = " << frobNorm(ptA_bar) << std::endl;

        // Step 4: Y = M^T / trace(M*M^T)
        auto encY = cc->EvalMult(encMt, encInvTr);
        auto ptY = ptScale(ptMt, ptInvTr);

        std::cout << "\nY_0 = M^T / trace(M*M^T):" << std::endl;
        printSample("  pt Y", ptY);
        printSample("  enc Y", decrypt(encY));
        std::cout << "  maxErr: " << std::scientific << maxError(decrypt(encY), ptY) << std::endl;

        // Iteration
        std::cout << "\n--- Iterations: Y = Y * (I + A_bar), A_bar = A_bar^2 ---" << std::endl;
        for (int iter = 0; iter < r; iter++) {
            int levelY = encY->GetLevel();
            int levelA = encA_bar->GetLevel();

            // Plaintext iteration
            auto ptI_A = ptAdd(I, ptA_bar);
            ptY = ptMult(ptY, ptI_A);
            ptA_bar = ptMult(ptA_bar, ptA_bar);

            // Bootstrap check
            bool bootstrapped = false;
            if (levelY >= multDepth - 2 || levelA >= multDepth - 2) {
                encA_bar = cc->EvalBootstrap(encA_bar, 2, 17);  // precision=17
                encY = cc->EvalBootstrap(encY, 2, 17);
                bootstrapped = true;
            }

            // Encrypted iteration - create plaintext at matching level
            auto pI_level = cc->MakeCKKSPackedPlaintext(I, 1, encA_bar->GetLevel(), nullptr, d * d);
            auto encI_A = cc->EvalAdd(pI_level, encA_bar);
            encY = matMult->eval_mult(encY, encI_A);
            encA_bar = matMult->eval_mult(encA_bar, encA_bar);

            // Compare
            auto decY = decrypt(encY);
            auto decA = decrypt(encA_bar);
            double errY = maxError(decY, ptY);
            double errA = maxError(decA, ptA_bar);

            std::cout << "iter " << std::setw(2) << iter
                      << " | lvl=" << levelY
                      << (bootstrapped ? " [BS]" : "     ")
                      << " | Y_err=" << std::scientific << std::setprecision(2) << errY
                      << " | A_err=" << errA
                      << " | ||A||=" << std::setprecision(2) << frobNorm(ptA_bar)
                      << std::endl;

            // Print sample values at key iterations
            if (iter == 0 || iter == 7 || iter == 8 || errY > 0.1) {
                printSample("    pt Y", ptY);
                printSample("    enc Y", decY);
                printSample("    pt A", ptA_bar);
                printSample("    enc A", decA);
            }

            if (errY > 1.0 || errA > 1.0) {
                std::cout << "*** ERROR EXPLODED! Stopping. ***" << std::endl;
                break;
            }
        }

        // Final result comparison
        auto groundTruth = computeGroundTruthInverse(matrix, d);
        auto result = decrypt(encY);
        std::cout << "\nFinal inverse maxErr: " << maxError(result, groundTruth) << std::endl;
        printSample("Ground truth M^{-1}", groundTruth);
        printSample("Computed", result);
    }
};

int main() {
    DebugIteration dbg;
    dbg.run();
    return 0;
}
