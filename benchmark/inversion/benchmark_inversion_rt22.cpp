#include "benchmark/benchmark.h"
#include "matrix_inversion_algo.h"
#include "matrix_utils.h"

const int ITERATION_COUNT = 1;

template <int d>
static void BM_RT22_Inversion(benchmark::State& state) {
    int multDepth;
    uint32_t scaleModSize;
    uint32_t firstModSize;
    int r;

    std::vector<uint32_t> levelBudget;
    std::vector<uint32_t> bsgsDim;

    CCParams<CryptoContextCKKSRNS> parameters;

    switch (d) {
    case 4:
        r = 13;
        multDepth = 2 * r + 12;
        scaleModSize = 50;
        break;
    case 8:
        r = 17;
        multDepth = 2 * r + 12;
        scaleModSize = 50;
        break;
    case 16:
        r = 20;
        multDepth = 37;
        scaleModSize = 59;
        firstModSize = 60;
        parameters.SetFirstModSize(firstModSize);
        levelBudget = {4, 4};
        bsgsDim = {0, 0};
        break;
    case 32:
        r = 22;
        multDepth = 37;
        scaleModSize = 59;
        firstModSize = 60;
        parameters.SetFirstModSize(firstModSize);
        levelBudget = {4, 4};
        bsgsDim = {0, 0};
        break;
    default:
        r = -1;
    }

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(d * d * d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();

    if (d >= 16) {
        cc->Enable(FHE);
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, d * d);
        cc->EvalBootstrapKeyGen(keyPair.secretKey, d * d); 
        // SetSlots(d*d) before bootstrapping
    }

    std::vector<int> rotations;
    for (int i = 1; i < d * d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matInv = std::make_unique<MatrixInverse_RT22<d>>(
        enc, cc, keyPair.publicKey, rotations, r, multDepth);

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) {
            matrix[i] = dis(gen);
        }
    } while (!utils::isInvertible(matrix, d));

    auto enc_matrix = enc->encryptInput(matrix);
    
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

BENCHMARK_TEMPLATE(BM_RT22_Inversion, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Inversion, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Inversion, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Inversion, 32)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);

BENCHMARK_MAIN();