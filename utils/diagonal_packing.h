#pragma once

#include <vector>

namespace utils {
    template <typename T>
    std::vector<std::vector<double>>
    extractDiagonalVectors(const std::vector<double> &packedMatrix, T size);

    template <typename T>
    std::vector<double>
    packDiagonalVectors(const std::vector<std::vector<double>> &diagonalVectors,
                        T size);
}

#include "diagonal_packing.hpp"  