#include "benchmark/benchmark.h"
#include "matrix_algo_multiPack.h"
#include "diagonal_packing.h"
#include "matrix_utils.h"
#include "encryption.h" 
#include "rotation.h"

using namespace lbcrypto;
const int ITERATION_COUNT = 1;

template <int d>
static void BM_Diag_Inversion(benchmark::State& state) {
    int r;
    switch (d) {
    case 4:
        r = 18;
        break;
    case 8:
        r = 21;
        break;
    case 16:
        r = 25;
        break;
    case 32:
        r = 28;
        break;
    case 64:
        r = 31;
        break;
    default:
        r = -1;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(r + 9);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d);  
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInv_diag<d>>(
        enc, cc, keyPair.publicKey, rotations, r);

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    do {
        // Generate diagonal dominant matrix
        for (size_t i = 0; i < d; i++) {
            double sum = 0;
            for (size_t j = 0; j < d; j++) {
                if (i != j) {
                    matrix[i * d + j] = dis(gen) * 0.1; // Off-diagonal elements
                    sum += std::abs(matrix[i * d + j]);
                }
            }
            matrix[i * d + i] = sum + dis(gen); // Make diagonal dominant
        }
    } while (!utils::isDiagonalMatrixInvertible(matrix, d));

    auto diagonals = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto& diag : diagonals) {
        enc_matrix.push_back(enc->encryptInput(diag));
    }
    
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matInv->eval_inverse(enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        state.counters["Time"] = duration;
        state.counters["MatrixSize"] = d;
        
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK_TEMPLATE(BM_Diag_Inversion, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_Diag_Inversion, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_Diag_Inversion, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);

BENCHMARK_MAIN();