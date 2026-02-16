// Test transpose accuracy for d=32 with mod fix - all algorithms
#include "matrix_algo_singlePack.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace lbcrypto;
using namespace BenchmarkConfig;

constexpr int d = 32;

template<typename MatOp>
double testTranspose(MatOp& matOp, CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keyPair,
                     const std::vector<double>& matrix, const std::vector<double>& pt_transpose, int batchSize) {
    auto pt_matrix = cc->MakeCKKSPackedPlaintext(matrix, 1, 0, nullptr, batchSize);
    auto enc_matrix = cc->Encrypt(keyPair.publicKey, pt_matrix);
    auto enc_transpose = matOp.eval_transpose(enc_matrix);

    Plaintext decrypted;
    cc->Decrypt(keyPair.secretKey, enc_transpose, &decrypted);
    decrypted->SetLength(d * d);
    auto result = decrypted->GetRealPackedValue();
    result.resize(d * d);

    double maxErr = 0.0;
    for (int i = 0; i < d * d; i++) {
        maxErr = std::max(maxErr, std::abs(result[i] - pt_transpose[i]));
    }
    return maxErr;
}

int main() {
    std::cout << "========== Transpose Test d=32 - All Algorithms ==========" << std::endl;

    int batchSize = d * d;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(32);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < batchSize; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);

    // Create random matrix
    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    // Compute plaintext transpose
    std::vector<double> pt_transpose(d * d);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            pt_transpose[j * d + i] = matrix[i * d + j];
        }
    }

    // Test JKLS18
    std::cout << "JKLS18: ";
    MatrixMult_JKLS18<d> jkls18(enc, cc, keyPair.publicKey, rotations);
    double err = testTranspose(jkls18, cc, keyPair, matrix, pt_transpose, batchSize);
    std::cout << "maxErr=" << std::scientific << err << ", log2=" << std::log2(err) << std::endl;

    // Test RT22
    std::cout << "RT22:   ";
    MatrixMult_RT22<d> rt22(enc, cc, keyPair.publicKey, rotations);
    err = testTranspose(rt22, cc, keyPair, matrix, pt_transpose, batchSize);
    std::cout << "maxErr=" << std::scientific << err << ", log2=" << std::log2(err) << std::endl;

    // Test AR24
    std::cout << "AR24:   ";
    MatrixMult_AR24<d> ar24(enc, cc, keyPair.publicKey, rotations);
    err = testTranspose(ar24, cc, keyPair, matrix, pt_transpose, batchSize);
    std::cout << "maxErr=" << std::scientific << err << ", log2=" << std::log2(err) << std::endl;

    // Test NewCol
    std::cout << "NewCol: ";
    MatrixMult_newCol<d> newcol(enc, cc, keyPair.publicKey, rotations);
    err = testTranspose(newcol, cc, keyPair, matrix, pt_transpose, batchSize);
    std::cout << "maxErr=" << std::scientific << err << ", log2=" << std::log2(err) << std::endl;

    std::cout << "\n========== Done ==========" << std::endl;
    return 0;
}
