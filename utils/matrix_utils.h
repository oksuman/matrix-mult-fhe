// utils/matrix_utils.h
#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace utils {

// Compute determinant using LU decomposition
double computeDeterminant(const std::vector<double>& matrix, int d) {
    std::vector<double> LU = matrix;
    double det = 1.0;

    for (int i = 0; i < d; i++) {
        // Partial pivoting
        int maxRow = i;
        double maxVal = std::abs(LU[i * d + i]);
        for (int k = i + 1; k < d; k++) {
            if (std::abs(LU[k * d + i]) > maxVal) {
                maxVal = std::abs(LU[k * d + i]);
                maxRow = k;
            }
        }

        if (maxVal < 1e-15) return 0.0;  // Singular

        if (maxRow != i) {
            for (int j = 0; j < d; j++) {
                std::swap(LU[i * d + j], LU[maxRow * d + j]);
            }
            det *= -1;  // Row swap changes sign
        }

        det *= LU[i * d + i];

        for (int j = i + 1; j < d; j++) {
            double factor = LU[j * d + i] / LU[i * d + i];
            for (int k = i; k < d; k++) {
                LU[j * d + k] -= factor * LU[i * d + k];
            }
        }
    }
    return det;
}

bool isInvertible(const std::vector<double>& matrix, int d) {
    // Check 1: Element condition (original check)
    double max_elem = 0;
    double min_elem = std::numeric_limits<double>::max();
    for (size_t i = 0; i < d * d; i++) {
        max_elem = std::max(max_elem, std::abs(matrix[i]));
        min_elem = std::min(min_elem, std::abs(matrix[i]));
    }
    if (min_elem < 1e-6 || max_elem / min_elem > 1e6) {
        return false;
    }

    // Check 2: Determinant check - ensure |det(M)| > threshold
    double det = computeDeterminant(matrix, d);
    double detThreshold = 0.1;
    if (std::abs(det) < detThreshold) {
        return false;
    }

    return true;
}

bool isDiagonalMatrixInvertible(const std::vector<double>& matrix, int d) {
    double max_elem = 0;
    double min_elem = std::numeric_limits<double>::max();
    
    for (int i = 0; i < d; i++) {
        double diag_elem = std::abs(matrix[i * d + i]);
        max_elem = std::max(max_elem, diag_elem);
        min_elem = std::min(min_elem, diag_elem);
    }

    if (min_elem < 1e-6 || max_elem / min_elem > 1e6) {
        return false;
    }
    return true;
}

} 