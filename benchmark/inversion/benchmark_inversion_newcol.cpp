// Matrix Inversion Benchmark - NewCol Algorithm
// Measures: Time (seconds) and Accuracy (Frobenius norm, log2 error)

#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>

using namespace lbcrypto;
using namespace BenchmarkConfig;

// Plaintext helper functions
template<int d>
std::vector<double> ptTranspose(const std::vector<double>& M) {
    std::vector<double> T(d*d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            T[j*d+i] = M[i*d+j];
    return T;
}

template<int d>
std::vector<double> ptMult(const std::vector<double>& A, const std::vector<double>& B) {
    std::vector<double> C(d*d, 0.0);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++)
                C[i*d+j] += A[i*d+k] * B[k*d+j];
    return C;
}

template<int d>
double ptTrace(const std::vector<double>& M) {
    double tr = 0.0;
    for (int i = 0; i < d; i++) tr += M[i*d+i];
    return tr;
}

template<int d>
std::vector<double> ptScale(const std::vector<double>& M, double s) {
    std::vector<double> R(d*d);
    for (int i = 0; i < d*d; i++) R[i] = M[i] * s;
    return R;
}

template<int d>
std::vector<double> ptAdd(const std::vector<double>& A, const std::vector<double>& B) {
    std::vector<double> C(d*d);
    for (int i = 0; i < d*d; i++) C[i] = A[i] + B[i];
    return C;
}

double maxErr(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); i++) err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

template <int d>
void runDebugInversion() {
    int scalarInvIter = getScalarInvIterations(d);
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;

    std::cout << "\n========== DEBUG NewCol d=" << d << " scalar_iter=" << scalarInvIter << " r=" << r << " ==========" << std::endl;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(d * d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

    std::vector<int> rotations;
    for (int i = 1; i < d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matMult = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

    // Generate matrix
    std::vector<double> M(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (int i = 0; i < d * d; i++) M[i] = dis(gen);
    } while (!utils::isInvertible(M, d));

    std::vector<double> I(d * d, 0.0);
    for (int i = 0; i < d; i++) I[i * d + i] = 1.0;

    // Decrypt helper
    auto decrypt = [&](const Ciphertext<DCRTPoly>& ct) {
        Plaintext pt;
        cc->Decrypt(keyPair.secretKey, ct, &pt);
        pt->SetLength(d * d);
        auto v = pt->GetRealPackedValue();
        v.resize(d * d);
        return v;
    };

    // Encrypt M
    auto ptM = cc->MakeCKKSPackedPlaintext(M, 1, 0, nullptr, d * d);
    auto encM = cc->Encrypt(keyPair.publicKey, ptM);
    std::cout << "[Input] lvl=" << encM->GetLevel() << std::endl;

    // Step 1: Transpose
    auto ptMt = ptTranspose<d>(M);
    auto encMt = matMult->eval_transpose(encM);
    std::cout << "[Transpose] lvl=" << encMt->GetLevel() << " maxErr=" << std::scientific << maxErr(decrypt(encMt), ptMt) << std::endl;

    // Step 2: M * M^T
    auto ptMMt = ptMult<d>(M, ptMt);
    auto encMMt = matMult->eval_mult(encM, encMt);
    std::cout << "[M*M^T] lvl=" << encMMt->GetLevel() << " maxErr=" << maxErr(decrypt(encMMt), ptMMt) << std::endl;

    // Step 3: trace
    double ptTr = ptTrace<d>(ptMMt);
    auto encTr = matMult->eval_trace(encMMt, d * d);
    auto decTr = decrypt(encTr);
    std::cout << "[Trace] lvl=" << encTr->GetLevel() << " pt=" << std::fixed << std::setprecision(4) << ptTr << " enc=" << decTr[0] << " err=" << std::scientific << std::abs(decTr[0] - ptTr) << std::endl;

    // Step 4: scalar inverse
    if ((int)encTr->GetLevel() >= multDepth - 10) {
        encTr = cc->EvalBootstrap(encTr, 2, 18);
        std::cout << "[Trace BS] lvl=" << encTr->GetLevel() << std::endl;
    }
    double ptInvTr = 1.0 / ptTr;
    auto encInvTr = matMult->eval_scalar_inverse(encTr, d * d, scalarInvIter, d * d);
    auto decInvTr = decrypt(encInvTr);
    std::cout << "[1/Trace] lvl=" << encInvTr->GetLevel() << " pt=" << std::fixed << std::setprecision(6) << ptInvTr << " enc=" << decInvTr[0] << " err=" << std::scientific << std::abs(decInvTr[0] - ptInvTr) << std::endl;

    // Step 5: Y_0, A_bar_0
    auto ptY = ptScale<d>(ptMt, ptInvTr);
    auto ptA = ptAdd<d>(I, ptScale<d>(ptMMt, -ptInvTr));
    Plaintext pI = cc->MakeCKKSPackedPlaintext(I, 1, 0, nullptr, d * d);
    auto encY = cc->EvalMultAndRelinearize(encMt, encInvTr);
    auto encA = cc->EvalSub(pI, cc->EvalMultAndRelinearize(encMMt, encInvTr));
    std::cout << "[Y_0] lvl=" << encY->GetLevel() << " maxErr=" << maxErr(decrypt(encY), ptY) << std::endl;
    std::cout << "[A_0] lvl=" << encA->GetLevel() << " maxErr=" << maxErr(decrypt(encA), ptA) << std::endl;

    // Iterations
    int bsCount = 0;
    for (int i = 0; i < r; i++) {
        bool bs = false;
        if ((int)encY->GetLevel() >= multDepth - 2) {
            encY = cc->EvalBootstrap(encY, 2, 18);
            encA = cc->EvalBootstrap(encA, 2, 18);
            bsCount += 2;
            bs = true;
        }
        auto ptIA = ptAdd<d>(I, ptA);
        ptY = ptMult<d>(ptY, ptIA);
        ptA = ptMult<d>(ptA, ptA);

        auto pI_lvl = cc->MakeCKKSPackedPlaintext(I, 1, encA->GetLevel(), nullptr, d * d);
        encY = matMult->eval_mult(encY, cc->EvalAdd(pI_lvl, encA));
        auto encA_copy = encA->Clone();
        encA = matMult->eval_mult(encA, encA_copy);

        std::cout << "[iter " << std::setw(2) << i << "] lvl=" << encY->GetLevel()
                  << (bs ? " [BS]" : "     ")
                  << " Y_err=" << std::scientific << maxErr(decrypt(encY), ptY)
                  << " A_err=" << maxErr(decrypt(encA), ptA) << std::endl;
    }
    std::cout << "[Done] Bootstrap count: " << bsCount << std::endl;

    // Final error
    auto groundTruth = computeGroundTruthInverse(M, d);
    auto result = decrypt(encY);
    std::cout << "[Final] maxErr vs inverse: " << maxErr(result, groundTruth) << std::endl;
}

template <int d>
void runInversionBenchmark(int numRuns = 1) {
    int scalarInvIter = getScalarInvIterations(d);
    std::cout << "\n========== NewCol Inversion d=" << d << " (scalar_iter=" << scalarInvIter << ") ==========" << std::endl;

    // Unified parameters
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    uint32_t scaleModSize = 59;
    uint32_t firstModSize = 60;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);

    int max_batch = 1 << 16;
    int s = std::min(max_batch / d / d, d);
    int batchSize = d * d;
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    cc->Enable(FHE);
    cc->EvalBootstrapSetup(levelBudget, bsgsDim, batchSize);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, batchSize);

    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_newCol<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth, scalarInvIter);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Iterations: " << r << std::endl;
    std::cout << "Mult Depth: " << multDepth << std::endl;
    std::cout << "Scalar Inv Iter: " << scalarInvIter << std::endl;

    double totalTime = 0.0;
    ErrorMetrics avgError;
    std::vector<double> times;

    for (int run = 0; run < numRuns; run++) {
        // Generate random invertible matrix with different seed per run
        std::vector<double> matrix(d * d);
        std::mt19937 gen(42 + run);  // Different seed per trial
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        do {
            for (int i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));

        auto groundTruth = computeGroundTruthInverse(matrix, d);
        auto enc_matrix = enc->encryptInput(matrix);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        totalTime += duration;

        // Decrypt and compute error
        Plaintext ptx;
        cc->Decrypt(keyPair.secretKey, result, &ptx);
        ptx->SetLength(d * d);
        std::vector<double> computed = ptx->GetRealPackedValue();
        computed.resize(d * d);

        ErrorMetrics error;
        error.compute(groundTruth, computed, d);

        if (run == 0) {
            avgError = error;
        }

        std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(2)
                  << duration << "s, log2(err)=" << std::setprecision(1) << error.log2FrobError << std::endl;
    }

    // Summary
    double avgTime = totalTime / numRuns;
    double stdDev = 0.0;
    for (double t : times) {
        stdDev += (t - avgTime) * (t - avgTime);
    }
    stdDev = std::sqrt(stdDev / numRuns);

    std::cout << "\n--- Summary (d=" << d << ") ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << avgTime << "s";
    if (numRuns > 1) {
        std::cout << " ± " << stdDev << "s";
    }
    std::cout << std::endl;
    avgError.print();
}

int main(int argc, char* argv[]) {
    int numRuns = 1;
    if (argc > 1) {
        numRuns = std::atoi(argv[1]);
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  Matrix Inversion Benchmark - NewCol" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Runs per dimension: " << numRuns << std::endl;

    #ifdef _OPENMP
    // omp_set_num_threads(1);  // Commented for multi-thread quick test
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP: Not enabled (single thread)" << std::endl;
    #endif

    runDebugInversion<16>();

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
