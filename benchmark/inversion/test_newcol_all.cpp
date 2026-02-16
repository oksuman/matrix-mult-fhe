// Test newCol inversion for all dimensions with detailed iteration tracking
#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace lbcrypto;
using namespace BenchmarkConfig;

template<int d>
class DebugNewColInversion {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    std::shared_ptr<Encryption> enc;
    std::vector<double> matrix;
    std::vector<int> rotations;
    std::vector<double> I;
    int r;
    int scalarInvIter;
    int multDepth = 32;
    unsigned int seed;

public:
    DebugNewColInversion(unsigned int seed_ = 42) : r(getInversionIterations(d)), scalarInvIter(getScalarInvIterations(d)), seed(seed_) {
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

        I.resize(d * d, 0.0);
        for (int i = 0; i < d; i++) I[i * d + i] = 1.0;

        matrix.resize(d * d);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (size_t i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));
    }

    unsigned int getSeed() const { return seed; }

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

    void printMatrix(const std::string& name, const std::vector<double>& v, int rows = 3, int cols = 3) {
        std::cout << name << " (" << d << "x" << d << "):" << std::endl;
        for (int i = 0; i < std::min(rows, d); i++) {
            std::cout << "  [";
            for (int j = 0; j < std::min(cols, d); j++) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(9) << v[i * d + j];
                if (j < std::min(cols, d) - 1) std::cout << ", ";
            }
            if (cols < d) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (rows < d) std::cout << "  ..." << std::endl;
    }

    void run() {
        std::cout << "\n========== NewCol Inversion d=" << d << " r=" << r << " scalar_iter=" << scalarInvIter << " seed=" << seed << " ==========" << std::endl;

        auto matMult = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

        // Original matrix
        printMatrix("Input M", matrix);

        // Step 1: Transpose
        auto encM = enc->encryptInput(matrix);
        auto ptMt = ptTranspose(matrix);
        auto encMt = matMult->eval_transpose(encM);
        auto decMt = decrypt(encMt);
        std::cout << "\n[Step 1] Transpose M^T" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decMt, ptMt) << std::endl;

        // Step 2: M * M^T
        auto ptMMt = ptMult(matrix, ptMt);
        auto encMMt = matMult->eval_mult(encM, encMt);
        auto decMMt = decrypt(encMMt);
        std::cout << "\n[Step 2] M * M^T" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decMMt, ptMMt) << std::endl;
        printMatrix("  pt M*M^T", ptMMt);
        printMatrix("  enc M*M^T", decMMt);

        // Step 3: Trace
        double ptTr = ptTrace(ptMMt);
        auto encTrace = matMult->eval_trace(encMMt, d * d);
        auto decTrace = decrypt(encTrace);
        std::cout << "\n[Step 3] trace(M*M^T)" << std::endl;
        std::cout << "  pt trace  = " << std::fixed << std::setprecision(6) << ptTr << std::endl;
        std::cout << "  enc trace = " << decTrace[0] << std::endl;
        std::cout << "  error     = " << std::scientific << std::abs(decTrace[0] - ptTr) << std::endl;

        // Step 4: 1/trace
        double ptInvTr = 1.0 / ptTr;
        auto encInvTr = matMult->eval_scalar_inverse(encTrace, d * d, scalarInvIter, d * d);
        auto decInvTr = decrypt(encInvTr);
        std::cout << "\n[Step 4] 1/trace (Newton-Raphson, " << scalarInvIter << " iter, upper=" << d*d << ")" << std::endl;
        std::cout << "  pt 1/trace  = " << std::fixed << std::setprecision(8) << ptInvTr << std::endl;
        std::cout << "  enc 1/trace = " << decInvTr[0] << std::endl;
        std::cout << "  error       = " << std::scientific << std::abs(decInvTr[0] - ptInvTr) << std::endl;

        // Step 5: A_bar = I - (M*M^T)/trace
        auto pI = cc->MakeCKKSPackedPlaintext(I, 1, 0, nullptr, d * d);
        auto encA_bar = cc->EvalSub(pI, cc->EvalMult(encMMt, encInvTr));
        auto ptA_bar = ptAdd(I, ptScale(ptMMt, -ptInvTr));
        std::cout << "\n[Step 5] A_bar = I - (M*M^T)/trace" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decrypt(encA_bar), ptA_bar) << std::endl;
        std::cout << "  ||A_bar||_F = " << std::fixed << std::setprecision(4) << frobNorm(ptA_bar) << std::endl;

        // Step 6: Y = M^T / trace
        auto encY = cc->EvalMult(encMt, encInvTr);
        auto ptY = ptScale(ptMt, ptInvTr);
        std::cout << "\n[Step 6] Y_0 = M^T / trace" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decrypt(encY), ptY) << std::endl;

        // Iterations
        std::cout << "\n--- Iterations ---" << std::endl;
        std::cout << std::setw(5) << "iter" << " | " << std::setw(4) << "lvl" << " | "
                  << std::setw(6) << "BS" << " | " << std::setw(12) << "Y_maxErr" << " | "
                  << std::setw(12) << "A_maxErr" << " | " << std::setw(10) << "||A||_F" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (int iter = 0; iter < r; iter++) {
            int levelY = encY->GetLevel();

            // Plaintext iteration
            auto ptI_A = ptAdd(I, ptA_bar);
            ptY = ptMult(ptY, ptI_A);
            ptA_bar = ptMult(ptA_bar, ptA_bar);

            // Bootstrap check
            bool bootstrapped = false;
            if (levelY >= multDepth - 2) {
                encA_bar = cc->EvalBootstrap(encA_bar, 2, 18);  // precision=18
                encY = cc->EvalBootstrap(encY, 2, 18);
                bootstrapped = true;
            }

            // Encrypted iteration
            auto pI_level = cc->MakeCKKSPackedPlaintext(I, 1, encA_bar->GetLevel(), nullptr, d * d);
            auto encI_A = cc->EvalAdd(pI_level, encA_bar);
            encY = matMult->eval_mult(encY, encI_A);
            encA_bar = matMult->eval_mult(encA_bar, encA_bar);

            auto decY = decrypt(encY);
            auto decA = decrypt(encA_bar);
            double errY = maxError(decY, ptY);
            double errA = maxError(decA, ptA_bar);

            std::cout << std::setw(5) << iter << " | "
                      << std::setw(4) << levelY << " | "
                      << std::setw(6) << (bootstrapped ? "[BS]" : "") << " | "
                      << std::scientific << std::setprecision(2) << std::setw(12) << errY << " | "
                      << std::setw(12) << errA << " | "
                      << std::fixed << std::setprecision(4) << std::setw(10) << frobNorm(ptA_bar) << std::endl;

            if (errY > 1.0 || errA > 1.0) {
                std::cout << "*** ERROR EXPLODED! Stopping. ***" << std::endl;
                break;
            }
        }

        // Final comparison
        auto groundTruth = computeGroundTruthInverse(matrix, d);
        auto result = decrypt(encY);
        double finalErr = maxError(result, groundTruth);

        std::cout << "\n--- Final Result ---" << std::endl;
        std::cout << "  Final maxErr vs ground truth: " << std::scientific << finalErr << std::endl;
        printMatrix("  Ground truth M^{-1}", groundTruth);
        printMatrix("  Computed result", result);

        // Compute A * A^{-1} to verify
        auto verification = ptMult(matrix, result);
        std::cout << "\n  Verification M * M^{-1} (should be I):" << std::endl;
        printMatrix("    M * result", verification);
        double verifyErr = maxError(verification, I);
        std::cout << "    maxErr from I: " << std::scientific << verifyErr << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "  NewCol Inversion Debug - All Dimensions" << std::endl;
    std::cout << "  Usage: ./test_newcol_all [d] [seed] [trials]" << std::endl;
    std::cout << "============================================" << std::endl;

    #ifdef _OPENMP
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #endif

    int d_only = 0;
    unsigned int seed = 42;
    int trials = 1;

    if (argc > 1) {
        d_only = std::atoi(argv[1]);
    }
    if (argc > 2) {
        seed = std::atoi(argv[2]);
    }
    if (argc > 3) {
        trials = std::atoi(argv[3]);
    }

    std::cout << "d=" << (d_only == 0 ? "all" : std::to_string(d_only))
              << ", seed=" << seed << ", trials=" << trials << std::endl;

    for (int trial = 0; trial < trials; trial++) {
        unsigned int trial_seed = seed + trial;
        std::cout << "\n############### TRIAL " << (trial + 1) << "/" << trials << " ###############" << std::endl;

        if (d_only == 0 || d_only == 4) {
            DebugNewColInversion<4> dbg4(trial_seed);
            dbg4.run();
        }
        if (d_only == 0 || d_only == 8) {
            DebugNewColInversion<8> dbg8(trial_seed);
            dbg8.run();
        }
        if (d_only == 0 || d_only == 16) {
            DebugNewColInversion<16> dbg16(trial_seed);
            dbg16.run();
        }
        if (d_only == 0 || d_only == 32) {
            DebugNewColInversion<32> dbg32(trial_seed);
            dbg32.run();
        }
        if (d_only == 0 || d_only == 64) {
            DebugNewColInversion<64> dbg64(trial_seed);
            dbg64.run();
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Debug Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
