#pragma once
#include "diagonal_packing.h"

namespace utils {
    template <typename T>
    std::vector<std::vector<double>>
    extractDiagonalVectors(const std::vector<double> &packedMatrix, T size) {
        const size_t d = static_cast<size_t>(size);
        std::vector<std::vector<double>> diagonalVectors(
            d, std::vector<double>(d, 0.0));

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                size_t index = (j + i) % d + d * (j % d);
                diagonalVectors[i][j] = packedMatrix[index];
            }
        }
        return diagonalVectors;
    }

    template <typename T>
    std::vector<double>
    packDiagonalVectors(const std::vector<std::vector<double>> &diagonalVectors,
                        T size) {
        const size_t d = static_cast<size_t>(size);
        std::vector<double> packedMatrix(d * d, 0.0);

        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j < d; ++j) {
                size_t index = (j + i) % d + d * (j % d);
                packedMatrix[index] = diagonalVectors[i][j];
            }
        }
        return packedMatrix;
    }
}