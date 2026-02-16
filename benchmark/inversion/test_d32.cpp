// Quick test for d=32 inversion
#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace lbcrypto;
using namespace BenchmarkConfig;

int main() {
    constexpr int d = 32;
    int scalarInvIter = getScalarInvIterations(d);

    std::cout << "========== NewCol Inversion d=" << d << " ==========\n";

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

    std::cout << "Creating crypto context...\n";
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    std::cout << "Setting up bootstrapping...\n";
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

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << "\n";
    std::cout << "Iterations: " << r << "\n";
    std::cout << "Mult Depth: " << multDepth << "\n";

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) {
            matrix[i] = dis(gen);
        }
    } while (!utils::isInvertible(matrix, d));

    auto groundTruth = computeGroundTruthInverse(matrix, d);
    auto enc_matrix = enc->encryptInput(matrix);

    std::cout << "Starting inversion...\n";
    auto start = std::chrono::high_resolution_clock::now();
    auto result = matInv->eval_inverse(enc_matrix);
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Completed in " << duration << "s\n";

    Plaintext ptx;
    cc->Decrypt(keyPair.secretKey, result, &ptx);
    ptx->SetLength(d * d);
    std::vector<double> computed = ptx->GetRealPackedValue();
    computed.resize(d * d);

    ErrorMetrics error;
    error.compute(groundTruth, computed, d);
    error.print();

    return 0;
}
