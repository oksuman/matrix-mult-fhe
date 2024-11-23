#include "benchmark/benchmark.h"
#include "naive_inversion.h"
#include "matrix_utils.h"

const int ITERATION_COUNT = 1;

template <int d>
static void BM_Naive_Inversion(benchmark::State& state) {
    // Setup parameters based on matrix dimension
    int multDepth;
    int r;
    
    // Configure based on dimension
    switch (d) {
    case 4:
        r = 18;
        multDepth = r + 9;
        break;
    case 8:
        r = 21;
        multDepth = r + 9;
        break;
    case 16:
        r = 25;
        multDepth = r + 9;
        break;
    default:
        r = -1;
        multDepth = -1;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(1);  
    parameters.SetSecurityLevel(HEStd_128_classic);
    
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto matOp = std::make_unique<MatrixOperations<d>>(enc, cc, keyPair.publicKey);

    // Generate random invertible matrix
    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    do {
        for (size_t i = 0; i < d * d; i++) {
            matrix[i] = dis(gen);
        }
    } while (!utils::isInvertible(matrix, d));

    // Encrypt matrix
    std::vector<Ciphertext<DCRTPoly>> enc_matrix(d * d);
    for(int i = 0; i < d * d; i++) {
        std::vector<double> value = {matrix[i]};
        enc_matrix[i] = enc->encryptInput(value);
    }

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "Matrix Size: " << d << "x" << d << std::endl;

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = matOp->inverseMatrix(enc_matrix, r);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        state.counters["Time"] = duration;
        state.counters["MatrixSize"] = d;
        
        benchmark::DoNotOptimize(result);
    }
}

// Register benchmarks for different matrix sizes
BENCHMARK_TEMPLATE(BM_Naive_Inversion, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_Naive_Inversion, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_Naive_Inversion, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);

BENCHMARK_MAIN();