// Debug: Verify intermediate results of matrix inversion for all algorithms
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
class DebugInversion {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    std::shared_ptr<Encryption> enc;
    std::vector<double> matrix;
    std::vector<int> rotations;
    int scalarInvIter;
    int multDepth = 28;

public:
    DebugInversion() : scalarInvIter(getScalarInvIterations(d)) {
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

        matrix.resize(d * d);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (size_t i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));
    }

    std::vector<double> decrypt(const Ciphertext<DCRTPoly>& ct, int len) {
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, ct, &ptx);
        ptx->SetLength(len);
        auto result = ptx->GetRealPackedValue();
        result.resize(len);
        return result;
    }

    std::vector<double> plaintextTranspose(const std::vector<double>& M) {
        std::vector<double> Mt(d * d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                Mt[j * d + i] = M[i * d + j];
        return Mt;
    }

    std::vector<double> plaintextMult(const std::vector<double>& A, const std::vector<double>& B) {
        std::vector<double> C(d * d, 0.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                for (int k = 0; k < d; k++)
                    C[i * d + j] += A[i * d + k] * B[k * d + j];
        return C;
    }

    double plaintextTrace(const std::vector<double>& M) {
        double tr = 0.0;
        for (int i = 0; i < d; i++) tr += M[i * d + i];
        return tr;
    }

    double maxError(const std::vector<double>& a, const std::vector<double>& b, int len) {
        double err = 0.0;
        for (int i = 0; i < len; i++) err = std::max(err, std::abs(a[i] - b[i]));
        return err;
    }

    template<typename MatMultType>
    void debugAlgorithm(const std::string& name, int batchSize) {
        std::cout << "\n----- " << name << " d=" << d << " (scalar_iter=" << scalarInvIter << ") -----" << std::endl;

        auto matMult = std::make_unique<MatMultType>(enc, cc, keyPair.publicKey, rotations);
        auto encM = enc->encryptInput(matrix);

        // Transpose
        std::cout << "  Transpose: " << std::flush;
        auto encMt = matMult->eval_transpose(encM);
        auto decMt = decrypt(encMt, d * d);
        auto ptMt = plaintextTranspose(matrix);
        std::cout << "maxErr=" << std::scientific << std::setprecision(2) << maxError(decMt, ptMt, d * d) << std::endl;

        // M * M^T
        std::cout << "  M*M^T:     " << std::flush;
        auto encMMt = matMult->eval_mult(encM, encMt);
        auto decMMt = decrypt(encMMt, d * d);
        auto ptMMt = plaintextMult(matrix, ptMt);
        std::cout << "maxErr=" << std::scientific << std::setprecision(2) << maxError(decMMt, ptMMt, d * d) << std::endl;

        // Trace
        std::cout << "  Trace:     " << std::flush;
        auto encTrace = matMult->eval_trace(encMMt, batchSize);
        auto decTrace = decrypt(encTrace, batchSize);
        double ptTrace = plaintextTrace(ptMMt);
        std::cout << "enc=" << std::fixed << std::setprecision(4) << decTrace[0]
                  << " pt=" << ptTrace << " err=" << std::scientific << std::abs(decTrace[0] - ptTrace) << std::endl;

        // Scalar inverse
        std::cout << "  1/Trace:   " << std::flush;
        auto encInv = matMult->eval_scalar_inverse(encTrace, d * d, scalarInvIter, batchSize);
        auto decInv = decrypt(encInv, batchSize);
        double ptInv = 1.0 / ptTrace;
        std::cout << "enc=" << std::fixed << std::setprecision(6) << decInv[0]
                  << " pt=" << ptInv << " err=" << std::scientific << std::abs(decInv[0] - ptInv) << std::endl;

        std::cout << "  [Info] upperBound=" << (d*d) << " actualTrace=" << std::fixed << std::setprecision(2) << ptTrace << std::endl;
    }

    void runJKLS18() { debugAlgorithm<MatrixMult_JKLS18<d>>("JKLS18", d * d); }
    void runNewCol() { debugAlgorithm<MatrixMult_newCol<d>>("NewCol", d * d); }
};

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  Debug Inversion - All Algorithms" << std::endl;
    std::cout << "============================================" << std::endl;
    #ifdef _OPENMP
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #endif

    // d=4: JKLS18, NewCol
    std::cout << "\n========== d=4 ==========" << std::endl;
    { DebugInversion<4> dbg; dbg.runJKLS18(); dbg.runNewCol(); }

    // d=8: JKLS18, NewCol
    std::cout << "\n========== d=8 ==========" << std::endl;
    { DebugInversion<8> dbg; dbg.runJKLS18(); dbg.runNewCol(); }

    // d=16: JKLS18, NewCol
    std::cout << "\n========== d=16 ==========" << std::endl;
    { DebugInversion<16> dbg; dbg.runJKLS18(); dbg.runNewCol(); }

    // d=32: JKLS18, NewCol
    std::cout << "\n========== d=32 ==========" << std::endl;
    { DebugInversion<32> dbg; dbg.runJKLS18(); dbg.runNewCol(); }

    // d=64: JKLS18, NewCol
    std::cout << "\n========== d=64 ==========" << std::endl;
    { DebugInversion<64> dbg; dbg.runJKLS18(); dbg.runNewCol(); }

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Debug Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
