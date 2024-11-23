#pragma once

#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace utils {
    double calculateMSE(
        const std::vector<double>& weights,
        const std::vector<double>& features,
        const std::vector<double>& outcomes,
        const std::string& output_filename,
        size_t feature_dim = 8,
        size_t sample_dim = 64) {
        
        // Average weights across groups of 8
        std::vector<double> trained_weights;
        for(size_t i = 0; i < feature_dim; i++) {
            double w = 0;
            for(size_t j = 0; j < 8; j++) {
                w += weights[8*i + j];
            }
            trained_weights.push_back(w/8);
        }
        
        std::ofstream outputFile(output_filename);
        outputFile << "Trained Weights: ";
        for(const auto& w : trained_weights) {
            outputFile << std::fixed << std::setprecision(6) << w << " ";
        }
        outputFile << "\n\n";
        
        outputFile << "Number of test samples: " << sample_dim << "\n\n";
        outputFile << "Sample-wise Predictions:\n";
        outputFile << "Sample\tPredicted\tActual\t\tAbs Error\tSquared Error\n";
        outputFile << "--------------------------------------------------------\n";
        
        double mse = 0.0;
        
        for (size_t i = 0; i < sample_dim; i++) {
            double prediction = 0.0;
            std::vector<double> current_features(features.begin() + i * sample_dim, 
                                               features.begin() + i * sample_dim + feature_dim);
            
            for (size_t j = 0; j < feature_dim; j++) {
                prediction += trained_weights[j] * current_features[j];
            }
            
            double error = prediction - outcomes[i];
            double abs_error = std::abs(error);
            double squared_error = error * error;
            mse += squared_error;
            
            outputFile << i + 1 << "\t" 
                      << std::fixed << std::setprecision(6) 
                      << prediction << "\t\t"
                      << outcomes[i] << "\t\t"
                      << abs_error << "\t\t"
                      << squared_error << "\n";
        }
        
        mse /= sample_dim;
        
        outputFile << "\n--------------------------------------------------------\n";
        outputFile << "Mean Squared Error (MSE): " << mse << std::endl;
        outputFile.close();
        
        return mse;
    }

    double calculateMSE(
        const std::vector<double>& weights,
        const std::vector<double>& features,
        const std::vector<double>& outcomes,
        size_t feature_dim = 8,
        size_t sample_dim = 64) {
        
        // Average weights across groups of 8
        std::vector<double> trained_weights;
        for(size_t i = 0; i < feature_dim; i++) {
            double w = 0;
            for(size_t j = 0; j < 8; j++) {
                w += weights[8*i + j];
            }
            trained_weights.push_back(w/8);
        }
        
        double mse = 0.0;
        
        for (size_t i = 0; i < sample_dim; i++) {
            std::vector<double> current_features(features.begin() + i * sample_dim, 
                                               features.begin() + i * sample_dim + feature_dim);
            double prediction = 0.0;
            for (size_t j = 0; j < feature_dim; j++) {
                prediction += trained_weights[j] * current_features[j];
            }
            mse += std::pow(prediction - outcomes[i], 2);
        }
        
        return mse / sample_dim;
    }
}