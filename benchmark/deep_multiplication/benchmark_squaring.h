#include "benchmark/benchmark.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>
#include "encryption.h"
#include "matrix_algo_singlePack.h"
#include "matrix_algo_multiPack.h"
#include "diagonal_packing.h" 

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include <filesystem>

constexpr int SQUARING_ITERATIONS = 15;
constexpr int Scaling = 50;

size_t GetSerializedSize(const Ciphertext<DCRTPoly>& ct, const std::string& tmpFile) {
    Serial::SerializeToFile(tmpFile, ct, SerType::BINARY);
    size_t size = std::filesystem::file_size(tmpFile);
    std::filesystem::remove(tmpFile);
    return size;
}

template <int d>
auto setupSquaringJKLS18() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS*3); 
    parameters.SetScalingModSize(Scaling);
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

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }
    auto enc_matrix = enc->encryptInput(matrix);
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    size_t serSize = GetSerializedSize(enc_matrix, "tmp_ct_jkls18.bin");
    std::cout << "[JKLS18] Serialized ciphertext size: " << serSize / 1024.0 << " KB" << std::endl;

    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}

template <int d>
struct SquaringSetupOutputRT22 {
    CryptoContext<DCRTPoly> cc;
    std::unique_ptr<MatrixMult_RT22<d>> algo;
    Ciphertext<DCRTPoly> enc_matrix;
    std::vector<Ciphertext<DCRTPoly>> enc_split;  // For Strassen's algorithm
};

template <int d>
auto setupSquaringRT22() -> SquaringSetupOutputRT22<d> {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS*2);  
    parameters.SetScalingModSize(Scaling);
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

    SquaringSetupOutputRT22<d> output{cc, std::move(algo)};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    if (d == 32) {
        std::vector<double> matrix(d * d);
        for (auto& elem : matrix) elem = dis(gen);
        output.enc_matrix = enc->encryptInput(matrix);

        size_t serSize = GetSerializedSize(output.enc_matrix, "tmp_ct_rt22_matrix.bin");
        std::cout << "[RT22 32x32] Serialized ciphertext size: " << serSize / 1024.0 << " KB" << std::endl;

        std::vector<double> largeMatrix(4 * d * d);
        for (auto& elem : largeMatrix) elem = dis(gen);

        std::vector<std::vector<double>> split(4, std::vector<double>(d * d));
        int largeSize = 2 * d;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                split[0][i * d + j] = largeMatrix[i * largeSize + j];
                split[1][i * d + j] = largeMatrix[i * largeSize + (j + d)];
                split[2][i * d + j] = largeMatrix[(i + d) * largeSize + j];
                split[3][i * d + j] = largeMatrix[(i + d) * largeSize + (j + d)];
            }
        }

        for (size_t i = 0; i < 4; ++i) {
            auto ct = enc->encryptInput(split[i]);
            output.enc_split.push_back(ct);
            size_t splitSize = GetSerializedSize(ct, "tmp_ct_rt22_split" + std::to_string(i) + ".bin");
            std::cout << "[RT22 64x64 Split Block " << i << "] Serialized size: " << splitSize / 1024.0 << " KB" << std::endl;
        }

    } else {
        std::vector<double> matrix(d * d);
        for (auto& elem : matrix) elem = dis(gen);
        output.enc_matrix = enc->encryptInput(matrix);

        size_t serSize = GetSerializedSize(output.enc_matrix, "tmp_ct_rt22_matrix.bin");
        std::cout << "[RT22] Serialized ciphertext size: " << serSize / 1024.0 << " KB" << std::endl;
    }
    return output;
}

template <int d>
auto setupSquaringAR24() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS*3);  
    parameters.SetScalingModSize(Scaling);
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
    auto algo = std::make_unique<MatrixMult_AR24<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    auto enc_matrix = enc->encryptInput(matrix);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    size_t serSize = GetSerializedSize(enc_matrix, "tmp_ct_ar24.bin");
    std::cout << "[AR24] Serialized ciphertext size: " << serSize / 1024.0 << " KB" << std::endl;

    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}

template <int d>
auto setupSquaringNewCol() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS*2); 
    parameters.SetScalingModSize(Scaling);
    parameters.SetSecurityLevel(HEStd_128_classic);

    parameters.SetBatchSize(d * d);  // Initial batch size is d*d

    auto cc = GenCryptoContext(parameters);
    int max_batch = cc->GetRingDimension()/2;

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keyPair = cc->KeyGen();

    int s = std::min(max_batch / d / d, d);
    std::vector<int> rotations;
    for (int i = 1; i < d * d * s; i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }

    cc->EvalRotateKeyGen(keyPair.secretKey, rotations);
    cc->EvalMultKeyGen(keyPair.secretKey);

    auto enc = std::make_shared<Encryption>(cc, keyPair.publicKey);
    auto algo = std::make_unique<MatrixMult_newCol<d>>(enc, cc, keyPair.publicKey, rotations);

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    auto enc_matrix = enc->encryptInput(matrix);

    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    size_t serSize = GetSerializedSize(enc_matrix, "tmp_ct_newcol.bin");
    std::cout << "[NewCol] Serialized ciphertext size: " << serSize / 1024.0 << " KB" << std::endl;

    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}

template <int d>
auto setupSquaringNewRow() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS*2);  
    parameters.SetScalingModSize(Scaling);
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

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    auto enc_matrix = enc->encryptInput(matrix);
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}

template <int d>
auto setupSquaringDiag() {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(SQUARING_ITERATIONS);  
    parameters.SetScalingModSize(Scaling);
    parameters.SetBatchSize(d);  // Note: Uses d instead of d*d
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
    auto algo = std::make_unique<MatrixInv_diag<d>>(enc, cc, keyPair.publicKey, rotations, -1);

    std::vector<double> matrix(d * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < d * d; i++) {
        matrix[i] = dis(gen);
    }

    auto diagonals = utils::extractDiagonalVectors(matrix, d);
    std::vector<Ciphertext<DCRTPoly>> enc_matrix;
    for (const auto& diag : diagonals) {
        enc_matrix.push_back(enc->encryptInput(diag));
    }
    std::cout << "Ring Dimension: " << cc->GetRingDimension() << std::endl;
    return std::make_tuple(std::move(cc), std::move(algo), std::move(enc_matrix));
}


// Benchmark functions for each algorithm
template <int d>
static void BM_JKLS18_Squaring(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupSquaringJKLS18<d>();
    std::vector<double> iteration_times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto current = enc_matrix;
        std::vector<double> squaring_times;
        
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            auto squaring_start = std::chrono::high_resolution_clock::now();
            current = algo->eval_mult(current, current);
            auto squaring_end = std::chrono::high_resolution_clock::now();
            
            double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
            squaring_times.push_back(squaring_duration);
            state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();
        iteration_times.push_back(total_duration);
        
        benchmark::DoNotOptimize(current);
    }
    
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    state.counters["TotalTime"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_RT22_Squaring(benchmark::State& state) {
    auto setup = setupSquaringRT22<d>();
    
    if (d == 32) {
        std::vector<double> times_normal;
        std::vector<double> times_strassen;

        for (auto _ : state) {
            {
                auto current = setup.enc_matrix;
                auto start_normal = std::chrono::high_resolution_clock::now();
                
                std::vector<double> squaring_times;
                for (int i = 0; i < SQUARING_ITERATIONS; i++) {
                    auto squaring_start = std::chrono::high_resolution_clock::now();
                    current = setup.algo->eval_mult(current, current);
                    auto squaring_end = std::chrono::high_resolution_clock::now();
                    
                    double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
                    squaring_times.push_back(squaring_duration);
                    state.counters["32x32_Round_" + std::to_string(i+1)] = squaring_duration;
                }
                
                auto end_normal = std::chrono::high_resolution_clock::now();
                benchmark::DoNotOptimize(current);
                double duration_normal = std::chrono::duration<double>(end_normal - start_normal).count();
                times_normal.push_back(duration_normal);
            }

            {
                auto current_splits = setup.enc_split;
                auto start_strassen = std::chrono::high_resolution_clock::now();
                
                std::vector<double> squaring_times;
                for (int i = 0; i < SQUARING_ITERATIONS; i++) {
                    auto squaring_start = std::chrono::high_resolution_clock::now();
                    current_splits = setup.algo->eval_mult_strassen(current_splits, current_splits);
                    auto squaring_end = std::chrono::high_resolution_clock::now();
                    
                    double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
                    squaring_times.push_back(squaring_duration);
                    state.counters["64x64_Strassen_Round_" + std::to_string(i+1)] = squaring_duration;
                }
                
                auto end_strassen = std::chrono::high_resolution_clock::now();
                benchmark::DoNotOptimize(current_splits);
                double duration_strassen = std::chrono::duration<double>(end_strassen - start_strassen).count();
                times_strassen.push_back(duration_strassen);
            }
        }

        double avg_time_normal = std::accumulate(times_normal.begin(), times_normal.end(), 0.0) / times_normal.size();
        state.counters["Time_32x32"] = avg_time_normal;
        state.counters["MatrixSize_32x32"] = 32;

        double avg_time_strassen = std::accumulate(times_strassen.begin(), times_strassen.end(), 0.0) / times_strassen.size();
        state.counters["Time_64x64_Strassen"] = avg_time_strassen;
        state.counters["MatrixSize_64x64_Strassen"] = 64;
    } else {
        std::vector<double> iteration_times;
        
        for (auto _ : state) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto current = setup.enc_matrix;
            std::vector<double> squaring_times;
            
            for (int i = 0; i < SQUARING_ITERATIONS; i++) {
                auto squaring_start = std::chrono::high_resolution_clock::now();
                current = setup.algo->eval_mult(current, current);
                auto squaring_end = std::chrono::high_resolution_clock::now();
                
                double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
                squaring_times.push_back(squaring_duration);
                state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double total_duration = std::chrono::duration<double>(end - start).count();
            iteration_times.push_back(total_duration);
            
            benchmark::DoNotOptimize(current);
        }
        
        double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
        state.counters["TotalTime"] = avg_time;
        state.counters["MatrixSize"] = d;
    }
}

template <int d>
static void BM_AR24_Squaring(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupSquaringAR24<d>();
    std::vector<double> iteration_times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto current = enc_matrix;
        std::vector<double> squaring_times;
        
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            auto squaring_start = std::chrono::high_resolution_clock::now();
            current = algo->eval_mult_and_clean(current, current);
            auto squaring_end = std::chrono::high_resolution_clock::now();
            
            double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
            squaring_times.push_back(squaring_duration);
            state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();
        iteration_times.push_back(total_duration);
        
        benchmark::DoNotOptimize(current);
    }
    
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    state.counters["TotalTime"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_NewCol_Squaring(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupSquaringNewCol<d>();
    std::vector<double> iteration_times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto current = enc_matrix;
        std::vector<double> squaring_times;
        
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            auto squaring_start = std::chrono::high_resolution_clock::now();
            current = algo->eval_mult(current, current);
            auto squaring_end = std::chrono::high_resolution_clock::now();
            
            double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
            squaring_times.push_back(squaring_duration);
            state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();
        iteration_times.push_back(total_duration);
        
        benchmark::DoNotOptimize(current);
    }
    
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    state.counters["TotalTime"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_NewRow_Squaring(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupSquaringNewRow<d>();
    std::vector<double> iteration_times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto current = enc_matrix;
        std::vector<double> squaring_times;
        
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            auto squaring_start = std::chrono::high_resolution_clock::now();
            current = algo->eval_mult(current, current);
            auto squaring_end = std::chrono::high_resolution_clock::now();
            
            double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
            squaring_times.push_back(squaring_duration);
            state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();
        iteration_times.push_back(total_duration);
        
        benchmark::DoNotOptimize(current);
    }
    
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    state.counters["TotalTime"] = avg_time;
    state.counters["MatrixSize"] = d;
}

template <int d>
static void BM_Diag_Squaring(benchmark::State& state) {
    auto [cc, algo, enc_matrix] = setupSquaringDiag<d>();
    std::vector<double> iteration_times;
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto current = enc_matrix;
        std::vector<double> squaring_times;
        
        for (int i = 0; i < SQUARING_ITERATIONS; i++) {
            auto squaring_start = std::chrono::high_resolution_clock::now();
            current = algo->eval_mult(current, current);
            auto squaring_end = std::chrono::high_resolution_clock::now();
            
            double squaring_duration = std::chrono::duration<double>(squaring_end - squaring_start).count();
            squaring_times.push_back(squaring_duration);
            state.counters["Round_" + std::to_string(i+1)] = squaring_duration;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_duration = std::chrono::duration<double>(end - start).count();
        iteration_times.push_back(total_duration);
        
        benchmark::DoNotOptimize(current);
    }
    
    double avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    state.counters["TotalTime"] = avg_time;
    state.counters["MatrixSize"] = d;
}