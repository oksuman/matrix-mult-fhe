// Test d=32 for all algorithms
#include "matrix_inversion_algo.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace lbcrypto;
using namespace BenchmarkConfig;

constexpr int d = 32;

void testJKLS18() {
    std::cout << "\n========== JKLS18 d=32 ==========" << std::endl;
    int scalarInvIter = getScalarInvIterations(d);
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;

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
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

    std::vector<int> rotations;
    for (int i = 1; i < d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_JKLS18<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth, scalarInvIter);

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) matrix[i] = dis(gen);
    } while (!utils::isInvertible(matrix, d));

    auto enc_matrix = enc->encryptInput(matrix);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "  Time: " << duration << "s - SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  FAILED: " << e.what() << std::endl;
    }
}

void testRT22() {
    std::cout << "\n========== RT22 d=32 ==========" << std::endl;
    int scalarInvIter = getScalarInvIterations(d);
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    int batchSize = d * d * d;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_RT22<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth, scalarInvIter);

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) matrix[i] = dis(gen);
    } while (!utils::isInvertible(matrix, d));

    auto enc_matrix = enc->encryptInput(matrix);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "  Time: " << duration << "s - SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  FAILED: " << e.what() << std::endl;
    }
}

void testAR24() {
    std::cout << "\n========== AR24 d=32 ==========" << std::endl;
    int scalarInvIter = getScalarInvIterations(d);
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    int max_batch = 1 << 16;
    int s = std::min(max_batch / d / d, d);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(d * d * s);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_AR24<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth, scalarInvIter);

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) matrix[i] = dis(gen);
    } while (!utils::isInvertible(matrix, d));

    auto pt_matrix = cc->MakeCKKSPackedPlaintext(matrix, 1, 0, nullptr, d * d);
    auto enc_matrix = cc->Encrypt(keyPair.publicKey, pt_matrix);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "  Time: " << duration << "s - SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  FAILED: " << e.what() << std::endl;
    }
}

void testNewCol() {
    std::cout << "\n========== NewCol d=32 ==========" << std::endl;
    int scalarInvIter = getScalarInvIterations(d);
    int r = getInversionIterations(d);
    int multDepth = MULT_DEPTH;
    int max_batch = 1 << 16;
    int s = std::min(max_batch / d / d, d);

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
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, d * d);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d);

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

    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) matrix[i] = dis(gen);
    } while (!utils::isInvertible(matrix, d));

    auto enc_matrix = enc->encryptInput(matrix);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << "  Time: " << duration << "s - SUCCESS" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  FAILED: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "========== Testing d=32 for all algorithms ==========" << std::endl;

    testJKLS18();
    testRT22();
    testAR24();
    testNewCol();

    std::cout << "\n========== Done ==========" << std::endl;
    return 0;
}
