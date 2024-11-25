#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>

class MatrixTest {
private:
    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

    // Helper functions from the original implementation
    std::vector<double> initializeIdentityMatrix(int d) {
        std::vector<double> I(d * d, 0.0);
        for(int i = 0; i < d; i++) {
            I[i * d + i] = 1.0;
        }
        return I;
    }

    std::vector<double> transposeMatrix(const std::vector<double>& M, int d) {
        std::vector<double> T(d * d);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                T[j * d + i] = M[i * d + j];
            }
        }
        return T;
    }

    std::vector<double> matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, int d) {
        std::vector<double> C(d * d, 0.0);
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                for(int k = 0; k < d; k++) {
                    C[i * d + j] += A[i * d + k] * B[k * d + j];
                }
            }
        }
        return C;
    }

    double matrixTrace(const std::vector<double>& M, int d) {
        double trace = 0.0;
        for(int i = 0; i < d; i++) {
            trace += M[i * d + i];
        }
        return trace;
    }

    std::vector<double> multiplyVectorByConstant(const std::vector<double>& v, double c) {
        std::vector<double> result = v;
        for(double& val : result) {
            val *= c;
        }
        return result;
    }

    std::vector<double> matrixSubtract(const std::vector<double>& A, const std::vector<double>& B, int d) {
        std::vector<double> C(d * d);
        for(int i = 0; i < d * d; i++) {
            C[i] = A[i] - B[i];
        }
        return C;
    }

    std::vector<double> matrixAdd(const std::vector<double>& A, const std::vector<double>& B, int d) {
        std::vector<double> C(d * d);
        for(int i = 0; i < d * d; i++) {
            C[i] = A[i] + B[i];
        }
        return C;
    }

    // Exact inverse using Gauss-Jordan elimination
    std::vector<double> exactInverse(const std::vector<double>& M, int d) {
        std::vector<double> augmented(d * 2 * d);
        std::vector<double> inverse(d * d);
        
        // Create augmented matrix [M|I]
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                augmented[i * (2 * d) + j] = M[i * d + j];
                augmented[i * (2 * d) + d + j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gauss-Jordan elimination
        for(int i = 0; i < d; i++) {
            // Find pivot
            double pivot = augmented[i * (2 * d) + i];
            for(int j = 0; j < 2 * d; j++) {
                augmented[i * (2 * d) + j] /= pivot;
            }

            // Eliminate column
            for(int j = 0; j < d; j++) {
                if(i != j) {
                    double factor = augmented[j * (2 * d) + i];
                    for(int k = 0; k < 2 * d; k++) {
                        augmented[j * (2 * d) + k] -= factor * augmented[i * (2 * d) + k];
                    }
                }
            }
        }

        // Extract inverse matrix
        for(int i = 0; i < d; i++) {
            for(int j = 0; j < d; j++) {
                inverse[i * d + j] = augmented[i * (2 * d) + d + j];
            }
        }

        return inverse;
    }

    std::vector<double> iterative_inverse(const std::vector<double>& M, int d, int r) {
        std::vector<double> I = initializeIdentityMatrix(d);
        std::vector<double> At = transposeMatrix(M, d);
        std::vector<double> AAt = matrixMultiply(M, transposeMatrix(M, d), d);

        double trace = matrixTrace(AAt, d);

        std::vector<double> Y = multiplyVectorByConstant(At, 1/trace);
        std::vector<double> A_bar = matrixSubtract(I, multiplyVectorByConstant(AAt, 1/trace), d);

        for(int i = 0; i < r; i++) {
            Y = matrixMultiply(Y, matrixAdd(I, A_bar, d), d);
            A_bar = matrixMultiply(A_bar, A_bar, d);
        }

        return Y;
    }

    bool checkConvergence(const std::vector<double>& A, const std::vector<double>& B, int d) {
        for(int i = 0; i < d * d; i++) {
            if(std::abs(A[i] - B[i]) > 0.0001) {
                return false;
            }
        }
        return true;
    }

    std::vector<double> generateRandomMatrix(int d) {
        std::vector<double> M(d * d);
        for(int i = 0; i < d * d; i++) {
            M[i] = dis(gen);
        }
        return M;
    }

public:
    MatrixTest() : gen(rd()), dis(-1.0, 1.0) {}

void runConvergenceTest() {
        std::vector<int> dimensions = {4, 8, 16, 32, 64};
        int num_trials = 10000;
        double percentile_target = 0.95;  // 상위 1%를 포함하는 r값 선택 (99th percentile)
        
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "matrix_convergence_" << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S") << ".txt";
        
        std::ofstream result_file(ss.str());
        
        result_file << "Matrix Convergence Analysis Results\n";
        result_file << "====================================\n\n";
        result_file << "Number of trials per dimension: " << num_trials << "\n";
        result_file << "Target percentile: " << percentile_target * 100 << "%\n\n";

        // 결과를 저장할 벡터
        std::vector<std::pair<int, std::vector<int>>> all_results;

        for(int d : dimensions) {
            std::vector<int> convergence_iterations;
            
            std::cout << "Processing dimension " << d << "x" << d << "...\n";
            
            for(int trial = 0; trial < num_trials; trial++) {
                std::vector<double> M = generateRandomMatrix(d);
                std::vector<double> exact_inv = exactInverse(M, d);
                
                int left = 1, right = 100;
                int min_r = -1;
                
                while(left <= right) {
                    int mid = (left + right) / 2;
                    std::vector<double> iterative_inv = iterative_inverse(M, d, mid);
                    
                    if(checkConvergence(exact_inv, iterative_inv, d)) {
                        min_r = mid;
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                }

                if(min_r != -1) {
                    convergence_iterations.push_back(min_r);
                }
            }
            
            all_results.push_back({d, convergence_iterations});
            
            if(!convergence_iterations.empty()) {
                std::sort(convergence_iterations.begin(), convergence_iterations.end());
                
                int percentile_index = static_cast<int>(convergence_iterations.size() * percentile_target);
                int recommended_r = convergence_iterations[percentile_index];
                
                result_file << "Dimension " << d << "x" << d << ":\n";
                result_file << "Recommended r (" << percentile_target * 100 << "th percentile): " 
                           << recommended_r << "\n\n";
                
                // Distribution diagram
                result_file << "Distribution of r values:\n";
                std::vector<int> distribution(10, 0);  // 10개 구간으로 나눔
                int max_r = convergence_iterations.back();
                int min_r = convergence_iterations.front();
                int range = max_r - min_r + 1;
                int bucket_size = (range + 9) / 10;  // 올림 나눗셈
                
                for(int r : convergence_iterations) {
                    int bucket = (r - min_r) / bucket_size;
                    if(bucket >= 0 && bucket < 10) {
                        distribution[bucket]++;
                    }
                }

                // 최대 높이를 50칸으로 정규화
                int max_count = *std::max_element(distribution.begin(), distribution.end());
                int diagram_height = 20;  // 다이어그램 높이
                
                // 세로축 레이블과 다이어그램 출력
                std::vector<std::string> diagram(diagram_height, "");
                for(int i = 0; i < diagram_height; i++) {
                    int threshold = (max_count * (diagram_height - i - 1)) / diagram_height;
                    std::string row = "";
                    for(int count : distribution) {
                        if(count > threshold) {
                            row += "██";
                        } else {
                            row += "  ";
                        }
                    }
                    
                    // 왼쪽에 수치 표시
                    int value = (max_count * (diagram_height - i)) / diagram_height;
                    result_file << std::setw(5) << value << " |" << row << "\n";
                }
                
                // x축 그리기
                result_file << "      ";
                for(int i = 0; i < 10; i++) {
                    result_file << "--";
                }
                result_file << "\n";
                
                // x축 레이블
                result_file << "      ";
                for(int i = 0; i < 10; i++) {
                    result_file << std::setw(2) << min_r + i * bucket_size;
                }
                result_file << "\n\n";
                
                // 통계 정보 추가
                result_file << "Statistics:\n";
                result_file << "- Total successful trials: " << convergence_iterations.size() << "\n";
                result_file << "- Success rate: " 
                           << std::fixed << std::setprecision(2)
                           << (double)convergence_iterations.size() / num_trials * 100 << "%\n";
                result_file << "- Range of r: " << min_r << " to " << max_r << "\n\n";
            }
        }
        
        result_file.close();
        
        std::cout << "\nResults have been saved to " << ss.str() << std::endl;
    }
};

int main() {
    MatrixTest test;
    test.runConvergenceTest();
    return 0;
}