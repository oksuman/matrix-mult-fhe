// utils/matrix_utils.h
#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace utils {

bool isInvertible(const std::vector<double>& matrix, int d) {
    double max_elem = 0;
    double min_elem = std::numeric_limits<double>::max();
    for (size_t i = 0; i < d * d; i++) {
        max_elem = std::max(max_elem, std::abs(matrix[i]));
        min_elem = std::min(min_elem, std::abs(matrix[i]));
    }

    if (min_elem < 1e-6 || max_elem / min_elem > 1e6) {
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