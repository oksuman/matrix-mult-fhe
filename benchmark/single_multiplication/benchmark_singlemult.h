#include <algorithm>
#include <benchmark/benchmark.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <openfhe.h>
#include <random>
#include <vector>

#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "matrix_algo_multiPack.h"
#include "diagonal_packing.h"

using namespace lbcrypto;

template <int d>
auto setupJKLS18() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(3);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d * d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_JKLS18<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrixA(d * d);
    std::vector<double> matrixB(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrixA[i] = dis(gen);
        matrixB[i] = dis(gen);
    }

    auto enc_matrixA = enc->encryptInput(matrixA);
    auto enc_matrixB = enc->encryptInput(matrixB);
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), 
                          std::move(enc_matrixA), std::move(enc_matrixB));
}

template <int d>
struct SetupOutput {
    CryptoContext<DCRTPoly> cc;
    std::unique_ptr<MatrixMult_RT22<d>> algo;

    std::vector<Ciphertext<DCRTPoly>> enc_splitA;
    std::vector<Ciphertext<DCRTPoly>> enc_splitB;
    Ciphertext<DCRTPoly> enc_matrixA;
    Ciphertext<DCRTPoly> enc_matrixB;
};

template <int d>
auto setupRT22() -> SetupOutput<d> {
    // Original setup code
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetScalingModSize(50);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetBatchSize(d * d * d);
    
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    
    auto keyPair = cc->KeyGen();
    std::vector<int> rotations;
    for (int i = 1; i < d * d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_RT22<d>>(enc, cc, keyPair.publicKey, rotations);
    
    SetupOutput<d> output {cc, std::move(algo)};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    if (d == 16) {
        // Initialization for 16x16 case
        std::vector<double> matrixA(d * d), matrixB(d * d);
        for (auto& elem : matrixA) elem = dis(gen);
        for (auto& elem : matrixB) elem = dis(gen);
        
        output.enc_matrixA = enc->encryptInput(matrixA);
        output.enc_matrixB = enc->encryptInput(matrixB);
        
        // Initialization for Strassen case (32x32)
        std::vector<double> largeMatrixA(4 * d * d), largeMatrixB(4 * d * d);
        for (auto& elem : largeMatrixA) elem = dis(gen);
        for (auto& elem : largeMatrixB) elem = dis(gen);
        
        std::vector<std::vector<double>> splitA(4, std::vector<double>(d * d));
        std::vector<std::vector<double>> splitB(4, std::vector<double>(d * d));
        
        int largeSize = 2 * d, smallSize = d;
        for (int i = 0; i < smallSize; i++) {
            for (int j = 0; j < smallSize; j++) {
                splitA[0][i * smallSize + j] = largeMatrixA[i * largeSize + j];
                splitA[1][i * smallSize + j] = largeMatrixA[i * largeSize + (j + smallSize)];
                splitA[2][i * smallSize + j] = largeMatrixA[(i + smallSize) * largeSize + j];
                splitA[3][i * smallSize + j] = largeMatrixA[(i + smallSize) * largeSize + (j + smallSize)];
                
                splitB[0][i * smallSize + j] = largeMatrixB[i * largeSize + j];
                splitB[1][i * smallSize + j] = largeMatrixB[i * largeSize + (j + smallSize)];
                splitB[2][i * smallSize + j] = largeMatrixB[(i + smallSize) * largeSize + j];
                splitB[3][i * smallSize + j] = largeMatrixB[(i + smallSize) * largeSize + (j + smallSize)];
            }
        }
        
        for (int i = 0; i < 4; i++) {
            output.enc_splitA.push_back(enc->encryptInput(splitA[i]));
            output.enc_splitB.push_back(enc->encryptInput(splitB[i]));
        }
    } else {
        // Initialization for other cases
        std::vector<double> matrixA(d * d), matrixB(d * d);
        for (auto& elem : matrixA) elem = dis(gen);
        for (auto& elem : matrixB) elem = dis(gen);
        
        output.enc_matrixA = enc->encryptInput(matrixA);
        output.enc_matrixB = enc->encryptInput(matrixB);
    }
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return output;
}

template <int d>
auto setupAS24() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(30);
    parameters.SetScalingModSize(50);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    int max_batch = cc->GetRingDimension()/2;
    int s = std::min(max_batch / d / d, d);
    parameters.SetBatchSize(d * d * s);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_AS24<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrixA(d * d);
    std::vector<double> matrixB(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrixA[i] = dis(gen);
        matrixB[i] = dis(gen);
    }

    auto enc_matrixA = enc->encryptInput(matrixA);
    auto enc_matrixB = enc->encryptInput(matrixB);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), 
                          std::move(enc_matrixA), std::move(enc_matrixB));
}

template <int d>
auto setupNewCol() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(30);
    parameters.SetScalingModSize(50);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    int max_batch = cc->GetRingDimension()/2;
    int s = std::min(max_batch / d / d, d);
    parameters.SetBatchSize(d * d);  // Initial batch size is d*d

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {  // Generate rotation keys up to d*d*s
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrixA(d * d);
    std::vector<double> matrixB(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrixA[i] = dis(gen);
        matrixB[i] = dis(gen);
    }

    auto enc_matrixA = enc->encryptInput(matrixA);
    auto enc_matrixB = enc->encryptInput(matrixB);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), 
                          std::move(enc_matrixA), std::move(enc_matrixB));
}

template <int d>
auto setupNewRow() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(2);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d * d);
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d * d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_newRow<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrixA(d * d);
    std::vector<double> matrixB(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrixA[i] = dis(gen);
        matrixB[i] = dis(gen);
    }

    auto enc_matrixA = enc->encryptInput(matrixA);
    auto enc_matrixB = enc->encryptInput(matrixB);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), 
                          std::move(enc_matrixA), std::move(enc_matrixB));
}

template <int d>
auto setupDiag() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1); 
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(d); 
    parameters.SetSecurityLevel(HEStd_128_classic);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    std::vector<int> rotations;
    for (int i = 1; i < d; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixInv_diag<d>>(enc, cc, keyPair.publicKey, rotations, -1);  // -1 for multiplication-only mode

    // Generate random matrix
    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    // Extract diagonals
    auto diagonals = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto& diag : diagonals) {
        enc_matrix.push_back(enc->encryptInput(diag));
    }
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}


template <int d>
static void BM_JKLS18(benchmark::State& state) {
    auto [cc, algo, enc_matrixA, enc_matrixB] = setupJKLS18<d>();
    
    std::vector<double> times;
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = algo->eval_mult(enc_matrixA, enc_matrixB);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        benchmark::DoNotOptimize(result);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    state.counters["Time"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_RT22(benchmark::State& state) {
    if (d == 32 || d == 64) {
        state.counters["Time"] = -1;
        state.counters["MatrixSize"] = d;
        return;
    }

    auto setup = setupRT22<d>();
    std::vector<double> times_normal;
    std::vector<double> times_strassen;

    for (auto _ : state) {
        if (d == 16) {
            // Run eval_mult for 16x16 matrices
            auto start_normal = std::chrono::high_resolution_clock::now();
            auto result_normal = setup.algo->eval_mult(setup.enc_matrixA, setup.enc_matrixB);
            auto end_normal = std::chrono::high_resolution_clock::now();
            benchmark::DoNotOptimize(result_normal);
            double duration_normal = std::chrono::duration<double>(end_normal - start_normal).count();
            times_normal.push_back(duration_normal);

            // Run eval_mult_strassen for 32x32 matrices
            auto start_strassen = std::chrono::high_resolution_clock::now();
            auto result_strassen = setup.algo->eval_mult_strassen(setup.enc_splitA, setup.enc_splitB);
            auto end_strassen = std::chrono::high_resolution_clock::now();
            benchmark::DoNotOptimize(result_strassen);
            double duration_strassen = std::chrono::duration<double>(end_strassen - start_strassen).count();
            times_strassen.push_back(duration_strassen);
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = setup.algo->eval_mult(setup.enc_matrixA, setup.enc_matrixB);
            auto end = std::chrono::high_resolution_clock::now();
            benchmark::DoNotOptimize(result);
            double duration = std::chrono::duration<double>(end - start).count();
            times_normal.push_back(duration);
        }
    }

    if (d == 16) {
        double avg_time_normal = std::accumulate(times_normal.begin(), times_normal.end(), 0.0) / times_normal.size();
        state.counters["Time_16x16"] = avg_time_normal;
        state.counters["MatrixSize_16x16"] = 16;

        double avg_time_strassen = std::accumulate(times_strassen.begin(), times_strassen.end(), 0.0) / times_strassen.size();
        state.counters["Time_32x32_Strassen"] = avg_time_strassen;
        state.counters["MatrixSize_32x32_Strassen"] = 32;
    } else {
        double avg_time = std::accumulate(times_normal.begin(), times_normal.end(), 0.0) / times_normal.size();
        state.counters["Time"] = avg_time;
        state.counters["MatrixSize"] = d;
    }
}

template <int d>
static void BM_AS24(benchmark::State& state) {
    auto [cc, algo, enc_matrixA, enc_matrixB] = setupAS24<d>();
    std::vector<double> times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = algo->eval_mult(enc_matrixA, enc_matrixB);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        benchmark::DoNotOptimize(result);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    state.counters["Time"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_NewCol(benchmark::State& state) {
    auto [cc, algo, enc_matrixA, enc_matrixB] = setupNewCol<d>();
    std::vector<double> times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = algo->eval_mult(enc_matrixA, enc_matrixB);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        benchmark::DoNotOptimize(result);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    state.counters["Time"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_NewRow(benchmark::State& state) {
    auto [cc, algo, enc_matrixA, enc_matrixB] = setupNewRow<d>();
    std::vector<double> times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = algo->eval_mult(enc_matrixA, enc_matrixB);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        benchmark::DoNotOptimize(result);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    state.counters["Time"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_Diag(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupDiag<d>();
    std::vector<double> times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = algo->eval_mult(enc_matrix, enc_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double>(end - start).count();
        times.push_back(duration);
        benchmark::DoNotOptimize(result);
    }
    
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    state.counters["Time"] = avg_time;
    state.counters["MatrixSize"] = d;
}