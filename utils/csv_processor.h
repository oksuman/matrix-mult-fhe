#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

class CSVProcessor {
public:
    static void processDataset(const std::string& filename, 
                             std::vector<double>& features,
                             std::vector<double>& outcomes,
                             int feature_num,
                             int row_num) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Skip header
        std::string header;
        std::getline(file, header);

        // Process each row
        for (int row = 0; row < row_num; ++row) {
            std::string line;
            std::getline(file, line);
            std::stringstream ss(line);
            std::string cell;

            // Read features
            for (int col = 0; col < feature_num; ++col) {
                std::getline(ss, cell, ',');
                features.push_back(std::stod(cell));
            }
            // Pad with zeros
            for (int i = 0; i < row_num - feature_num; ++i) {
                features.push_back(0.0);
            }

            // Read outcome
            std::getline(ss, cell, ',');
            outcomes.push_back(std::stod(cell));
        }
        file.close();

        // Normalize features
        for (int col = 0; col < feature_num; ++col) {
            double max_value = features[col];
            for (int current_row = 1; current_row < row_num; ++current_row) {
                double current_value = features[col + current_row * row_num];
                if (current_value > max_value) {
                    max_value = current_value;
                }
            }

            if (max_value != 0.0) {
                for (int current_row = 0; current_row < row_num; ++current_row) {
                    features[col + current_row * row_num] /= max_value;
                }
            }
        }
    }
};