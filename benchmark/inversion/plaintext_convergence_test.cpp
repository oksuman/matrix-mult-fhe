// Plaintext convergence test - same conditions as benchmark
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <limits>

// Same as benchmark_config.h (95th percentile)
inline int getInversionIterations(int d) {
    switch(d) {
        case 4:  return 18;
        case 8:  return 22;
        case 16: return 25;
        case 32: return 27;
        case 64: return 31;
        default: return 25;
    }
}

// Same as matrix_utils.h
namespace utils {
double computeDeterminant(const std::vector<double>& matrix, int d) {
    std::vector<double> LU = matrix;
    double det = 1.0;
    for (int i = 0; i < d; i++) {
        int maxRow = i;
        double maxVal = std::abs(LU[i * d + i]);
        for (int k = i + 1; k < d; k++) {
            if (std::abs(LU[k * d + i]) > maxVal) {
                maxVal = std::abs(LU[k * d + i]);
                maxRow = k;
            }
        }
        if (maxVal < 1e-15) return 0.0;
        if (maxRow != i) {
            for (int j = 0; j < d; j++)
                std::swap(LU[i * d + j], LU[maxRow * d + j]);
            det *= -1;
        }
        det *= LU[i * d + i];
        for (int j = i + 1; j < d; j++) {
            double factor = LU[j * d + i] / LU[i * d + i];
            for (int k = i; k < d; k++)
                LU[j * d + k] -= factor * LU[i * d + k];
        }
    }
    return det;
}

bool isInvertible(const std::vector<double>& matrix, int d) {
    double max_elem = 0;
    double min_elem = std::numeric_limits<double>::max();
    for (size_t i = 0; i < d * d; i++) {
        max_elem = std::max(max_elem, std::abs(matrix[i]));
        min_elem = std::min(min_elem, std::abs(matrix[i]));
    }
    if (min_elem < 1e-6 || max_elem / min_elem > 1e6) return false;
    double det = computeDeterminant(matrix, d);
    if (std::abs(det) < 0.1) return false;
    return true;
}
}

std::vector<double> transposeMatrix(const std::vector<double>& M, int d) {
    std::vector<double> T(d * d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            T[j * d + i] = M[i * d + j];
    return T;
}

std::vector<double> matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, int d) {
    std::vector<double> C(d * d, 0.0);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++)
                C[i * d + j] += A[i * d + k] * B[k * d + j];
    return C;
}

double matrixTrace(const std::vector<double>& M, int d) {
    double trace = 0.0;
    for (int i = 0; i < d; i++) trace += M[i * d + i];
    return trace;
}

std::vector<double> matrixAdd(const std::vector<double>& A, const std::vector<double>& B, int d) {
    std::vector<double> C(d * d);
    for (int i = 0; i < d * d; i++) C[i] = A[i] + B[i];
    return C;
}

std::vector<double> matrixSub(const std::vector<double>& A, const std::vector<double>& B, int d) {
    std::vector<double> C(d * d);
    for (int i = 0; i < d * d; i++) C[i] = A[i] - B[i];
    return C;
}

std::vector<double> matrixScale(const std::vector<double>& A, double s, int d) {
    std::vector<double> C(d * d);
    for (int i = 0; i < d * d; i++) C[i] = A[i] * s;
    return C;
}

std::vector<double> exactInverse(const std::vector<double>& M, int d) {
    std::vector<double> aug(d * 2 * d);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            aug[i * (2 * d) + j] = M[i * d + j];
            aug[i * (2 * d) + d + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < d; i++) {
        double pivot = aug[i * (2 * d) + i];
        for (int j = 0; j < 2 * d; j++) aug[i * (2 * d) + j] /= pivot;
        for (int j = 0; j < d; j++) {
            if (i != j) {
                double factor = aug[j * (2 * d) + i];
                for (int k = 0; k < 2 * d; k++)
                    aug[j * (2 * d) + k] -= factor * aug[i * (2 * d) + k];
            }
        }
    }
    std::vector<double> inv(d * d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            inv[i * d + j] = aug[i * (2 * d) + d + j];
    return inv;
}

// Iterative inverse with specified r (same as FHE algorithm)
std::vector<double> iterativeInverse(const std::vector<double>& M, int d, int r) {
    std::vector<double> I(d * d, 0.0);
    for (int i = 0; i < d; i++) I[i * d + i] = 1.0;

    auto Mt = transposeMatrix(M, d);
    auto MMt = matrixMultiply(M, Mt, d);
    double trace = matrixTrace(MMt, d);

    auto Y = matrixScale(Mt, 1.0 / trace, d);
    auto A_bar = matrixSub(I, matrixScale(MMt, 1.0 / trace, d), d);

    for (int i = 0; i < r; i++) {
        Y = matrixMultiply(Y, matrixAdd(I, A_bar, d), d);
        A_bar = matrixMultiply(A_bar, A_bar, d);
    }
    return Y;
}

double frobeniusNorm(const std::vector<double>& A, int d) {
    double sum = 0.0;
    for (int i = 0; i < d * d; i++) sum += A[i] * A[i];
    return std::sqrt(sum);
}

double relativeError(const std::vector<double>& computed, const std::vector<double>& exact, int d) {
    auto diff = matrixSub(computed, exact, d);
    return frobeniusNorm(diff, d) / frobeniusNorm(exact, d);
}

template<int d>
void testDimension(int numTrials) {
    int r = getInversionIterations(d);
    std::cout << "\n===== d=" << d << ", r=" << r << " =====" << std::endl;

    int success = 0;
    int fail = 0;

    for (int trial = 0; trial < numTrials; trial++) {
        // Same seed pattern as benchmark
        std::mt19937 gen(42 + trial);
        std::uniform_real_distribution<double> dis(-1.0, 1.0);

        std::vector<double> matrix(d * d);
        int attempts = 0;
        do {
            attempts++;
            for (int i = 0; i < d * d; i++) {
                matrix[i] = dis(gen);
            }
        } while (!utils::isInvertible(matrix, d));

        auto exact = exactInverse(matrix, d);
        auto iterative = iterativeInverse(matrix, d, r);
        double relErr = relativeError(iterative, exact, d);
        double log2Err = std::log2(relErr);

        bool converged = (log2Err < -10);  // threshold: log2(err) < -10

        std::cout << "  Trial " << (trial + 1) << " (seed=" << (42 + trial)
                  << ", attempts=" << attempts << "): log2(err)="
                  << std::fixed << std::setprecision(1) << log2Err
                  << (converged ? " OK" : " FAIL") << std::endl;

        if (converged) success++;
        else fail++;
    }

    std::cout << "  Result: " << success << "/" << numTrials << " converged" << std::endl;
}

int main(int argc, char* argv[]) {
    int numTrials = 10;
    if (argc > 1) numTrials = std::atoi(argv[1]);

    std::cout << "============================================" << std::endl;
    std::cout << "  Plaintext Convergence Test" << std::endl;
    std::cout << "  Same conditions as benchmark" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Trials per dimension: " << numTrials << std::endl;

    testDimension<4>(numTrials);
    testDimension<8>(numTrials);
    testDimension<16>(numTrials);
    testDimension<32>(numTrials);
    testDimension<64>(numTrials);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Test Complete" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
