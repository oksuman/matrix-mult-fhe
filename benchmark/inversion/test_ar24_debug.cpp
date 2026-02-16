// Debug AR24 inversion with intermediate comparison to plaintext
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
class DebugAR24Inversion {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    std::shared_ptr<Encryption> enc;
    std::vector<double> matrix;
    std::vector<int> rotations;
    std::vector<double> I;
    int r;
    int scalarInvIter;
    int multDepth = 32;
    int s;
    unsigned int seed;

public:
    DebugAR24Inversion(unsigned int seed_ = 42) : r(getInversionIterations(d)), scalarInvIter(getScalarInvIterations(d)), seed(seed_) {
        int max_batch = 1 << 16;
        s = std::min(max_batch / d / d, d);

        uint32_t scaleModSize = 59;
        uint32_t firstModSize = 60;
        std::vector<uint32_t> levelBudget = {4, 4};
        std::vector<uint32_t> bsgsDim = {0, 0};

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetScalingModSize(scaleModSize);
        parameters.SetFirstModSize(firstModSize);
        parameters.SetBatchSize(d * d * s);
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

        for (int i = 1; i < d * d * s; i *= 2) {
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

    std::vector<double> decrypt(const Ciphertext<DCRTPoly>& ct, int slots = d * d) {
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, ct, &ptx);
        ptx->SetLength(slots);
        auto result = ptx->GetRealPackedValue();
        result.resize(slots);
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

    double maxError(const std::vector<double>& a, const std::vector<double>& b, int len = d * d) {
        double err = 0.0;
        for (int i = 0; i < len; i++) err = std::max(err, std::abs(a[i] - b[i]));
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
        std::cout << "\n========== AR24 Inversion d=" << d << " s=" << s << " r=" << r << " seed=" << seed << " ==========" << std::endl;

        auto matMult = std::make_unique<MatrixMult_AR24<d>>(enc, cc, keyPair.publicKey, rotations);

        printMatrix("Input M", matrix);

        auto ptM = cc->MakeCKKSPackedPlaintext(matrix, 1, 0, nullptr, d * d);
        auto encM = cc->Encrypt(keyPair.publicKey, ptM);
        auto ptMt = ptTranspose(matrix);
        auto encMt = matMult->eval_transpose(encM);
        auto decMt = decrypt(encMt);
        std::cout << "\n[Step 1] Transpose M^T" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decMt, ptMt) << std::endl;

        auto encM_exp = encM->Clone();
        encM_exp->SetSlots(d * d * s);
        encM_exp = matMult->clean(encM_exp);
        auto encMt_exp = encMt->Clone();
        encMt_exp->SetSlots(d * d * s);
        encMt_exp = matMult->clean(encMt_exp);

        auto ptMMt = ptMult(matrix, ptMt);
        auto encMMt = matMult->eval_mult(encM_exp, encMt_exp);
        auto decMMt = decrypt(encMMt);
        std::cout << "\n[Step 2] M * M^T" << std::endl;
        std::cout << "  maxErr = " << std::scientific << maxError(decMMt, ptMMt) << std::endl;

        double ptTr = ptTrace(ptMMt);
        auto encTrace = matMult->eval_trace(encMMt, d * d);
        auto decTrace = decrypt(encTrace);
        std::cout << "\n[Step 3] trace(M*M^T)" << std::endl;
        std::cout << "  pt trace  = " << std::fixed << std::setprecision(6) << ptTr << std::endl;
        std::cout << "  enc trace = " << decTrace[0] << std::endl;
        std::cout << "  error     = " << std::scientific << std::abs(decTrace[0] - ptTr) << std::endl;

        // Step 4: 1/trace
        double ptInvTr = 1.0 / ptTr;
        auto encInvTr = matMult->eval_scalar_inverse(encTrace, d * d, scalarInvIter, d * d * s);
        auto decInvTr = decrypt(encInvTr);
        std::cout << "\n[Step 4] 1/trace" << std::endl;
        std::cout << "  pt 1/trace  = " << std::fixed << std::setprecision(8) << ptInvTr << std::endl;
        std::cout << "  enc 1/trace = " << decInvTr[0] << std::endl;
        std::cout << "  error       = " << std::scientific << std::abs(decInvTr[0] - ptInvTr) << std::endl;

        // Step 5: Initialize Y and A_bar
        Plaintext pI = cc->MakeCKKSPackedPlaintext(I);
        auto encY = cc->EvalMultAndRelinearize(encMt, encInvTr);
        auto encA_bar = cc->EvalSub(pI, cc->EvalMultAndRelinearize(encMMt, encInvTr));

        auto ptY = ptScale(ptMt, ptInvTr);
        auto ptA_bar = ptAdd(I, ptScale(ptMMt, -ptInvTr));

        std::cout << "\n[Step 5] Initialize Y_0 and A_bar_0 (d*d slots)" << std::endl;
        std::cout << "  Y maxErr     = " << std::scientific << maxError(decrypt(encY), ptY) << std::endl;
        std::cout << "  A_bar maxErr = " << std::scientific << maxError(decrypt(encA_bar), ptA_bar) << std::endl;
        std::cout << "  ||A_bar||_F  = " << std::fixed << std::setprecision(4) << frobNorm(ptA_bar) << std::endl;

        std::cout << "\n--- Iterations ---" << std::endl;
        std::cout << std::setw(5) << "iter" << " | " << std::setw(4) << "lvl" << " | "
                  << std::setw(6) << "BS" << " | " << std::setw(12) << "Y_maxErr" << " | "
                  << std::setw(12) << "A_maxErr" << " | " << std::setw(10) << "||A||_F" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        for (int iter = 0; iter < r; iter++) {
            int levelY = encY->GetLevel();

            auto ptI_A = ptAdd(I, ptA_bar);
            ptY = ptMult(ptY, ptI_A);
            ptA_bar = ptMult(ptA_bar, ptA_bar);

            bool bootstrapped = false;
            if (levelY >= multDepth - 3) {
                encY = cc->EvalBootstrap(encY, 2, 18);
                encA_bar = cc->EvalBootstrap(encA_bar, 2, 18);
                bootstrapped = true;
            }

            auto pI_level = cc->MakeCKKSPackedPlaintext(I, 1, encA_bar->GetLevel(), nullptr, d * d);
            auto encI_A = cc->EvalAdd(pI_level, encA_bar);

            encI_A->SetSlots(d * d * s);
            encI_A = matMult->clean(encI_A);
            encY->SetSlots(d * d * s);
            encY = matMult->clean(encY);
            encY = matMult->eval_mult(encY, encI_A);

            encA_bar->SetSlots(d * d * s);
            encA_bar = matMult->clean(encA_bar);
            auto encA_bar_copy = encA_bar->Clone();
            encA_bar = matMult->eval_mult(encA_bar, encA_bar_copy);

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

        // Verification
        auto verification = ptMult(matrix, result);
        std::cout << "\n  Verification M * M^{-1} (should be I):" << std::endl;
        double verifyErr = maxError(verification, I);
        std::cout << "    maxErr from I: " << std::scientific << verifyErr << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "  AR24 Inversion Debug" << std::endl;
    std::cout << "============================================" << std::endl;

    int d_only = 4;
    unsigned int seed = 42;

    if (argc > 1) d_only = std::atoi(argv[1]);
    if (argc > 2) seed = std::atoi(argv[2]);

    std::cout << "d=" << d_only << ", seed=" << seed << std::endl;

    if (d_only == 4) {
        DebugAR24Inversion<4> dbg(seed);
        dbg.run();
    } else if (d_only == 8) {
        DebugAR24Inversion<8> dbg(seed);
        dbg.run();
    } else if (d_only == 16) {
        DebugAR24Inversion<16> dbg(seed);
        dbg.run();
    }

    return 0;
}
