// Test RT22 trace with d*d vs d*d*d batchSize
#include "matrix_algo_singlePack.h"
#include "matrix_utils.h"
#include "../benchmark_config.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace lbcrypto;
using namespace BenchmarkConfig;

constexpr int d = 4;

int main() {
    std::cout << "========== RT22 Trace Test d=4 ==========" << std::endl;

    int multDepth = MULT_DEPTH;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    int batchSize = d * d * d;  // RT22 uses d^3
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
    MatrixMult_RT22<d> rt22(enc, cc, keyPair.publicKey, rotations);

    // Create random matrix
    std::vector<double> matrix(d * d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    // Compute plaintext trace(M^T * M)
    std::vector<double> Mt(d * d);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            Mt[j * d + i] = matrix[i * d + j];
        }
    }
    std::vector<double> MtM(d * d, 0.0);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                MtM[i * d + j] += matrix[i * d + k] * Mt[k * d + j];
            }
        }
    }
    double pt_trace = 0.0;
    for (int i = 0; i < d; i++) {
        pt_trace += MtM[i * d + i];
    }
    std::cout << "Plaintext trace(M*M^T): " << pt_trace << std::endl;

    // Encrypt and compute
    auto pt_matrix = cc->MakeCKKSPackedPlaintext(matrix, 1, 0, nullptr, d * d);
    auto enc_matrix = cc->Encrypt(keyPair.publicKey, pt_matrix);

    auto M_transposed = rt22.eval_transpose(enc_matrix);
    auto MM_transposed = rt22.eval_mult(enc_matrix, M_transposed);

    // Test 1: trace with d*d*d (original)
    std::cout << "\n--- Test 1: eval_trace with batchSize = d*d*d (" << d*d*d << ") ---" << std::endl;
    auto trace1 = rt22.eval_trace(MM_transposed, d * d * d);

    Plaintext decrypted1;
    cc->Decrypt(keyPair.secretKey, trace1, &decrypted1);
    decrypted1->SetLength(1);
    double result1 = decrypted1->GetRealPackedValue()[0];
    std::cout << "Encrypted trace: " << result1 << std::endl;
    std::cout << "Error: " << std::abs(result1 - pt_trace) << std::endl;

    // Test 2: SetSlots to d*d then trace with d*d
    std::cout << "\n--- Test 2: SetSlots(d*d) then eval_trace with batchSize = d*d (" << d*d << ") ---" << std::endl;
    auto MM_transposed_dd = MM_transposed->Clone();
    MM_transposed_dd->SetSlots(d * d);
    auto trace2 = rt22.eval_trace(MM_transposed_dd, d * d);

    Plaintext decrypted2;
    cc->Decrypt(keyPair.secretKey, trace2, &decrypted2);
    decrypted2->SetLength(1);
    double result2 = decrypted2->GetRealPackedValue()[0];
    std::cout << "Encrypted trace: " << result2 << std::endl;
    std::cout << "Error: " << std::abs(result2 - pt_trace) << std::endl;

    std::cout << "\n========== Done ==========" << std::endl;
    return 0;
}
