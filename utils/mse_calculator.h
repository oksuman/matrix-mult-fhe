// utils/mse_calculator.h
#pragma once

#include <vector>
#include <cmath>

namespace utils {
    double calculateMSE(const std::vector<double>& weights,
                    const std::vector<double>& features,
                    const std::vector<double>& outcomes,
                    size_t feature_dim = 8,
                    size_t sample_dim = 64) {
        double mse = 0.0;
        
        for (size_t i = 0; i < sample_dim; i++) {
            double prediction = 0.0;
            for (size_t j = 0; j < feature_dim; j++) {
                prediction += weights[j] * features[i * sample_dim + j];
            }
            mse += std::pow(prediction - outcomes[i], 2);
        }
        
        return mse / sample_dim;
    }

} 