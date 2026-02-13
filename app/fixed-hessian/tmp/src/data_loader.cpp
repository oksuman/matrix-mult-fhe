// src/data_loader.cpp
#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cctype>

Dataset::Dataset(int samples, int features) 
    : X(samples, features), y(samples), 
      n_samples(samples), n_features(features) {}

bool is_numeric(const std::string& str) {
    if (str.empty()) return false;
    
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    
    std::string trimmed = str.substr(start, end - start + 1);
    
    try {
        std::stod(trimmed);
        return true;
    } catch (...) {
        return false;
    }
}

Dataset DataLoader::load_csv(const std::string& filename, bool has_header) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    int line_number = 0;
    bool auto_detect_header = !has_header;
    
    if (std::getline(file, line)) {
        line_number++;
        
        std::stringstream ss(line);
        std::string first_value;
        std::getline(ss, first_value, ',');
        
        if (auto_detect_header && !is_numeric(first_value)) {
            std::cout << "Detected header in CSV file, skipping first line" << std::endl;
            has_header = true;
        } else {
            std::vector<double> row;
            std::stringstream ss2(line);
            std::string value;
            int col = 0;
            
            while (std::getline(ss2, value, ',')) {
                col++;
                size_t start = value.find_first_not_of(" \t\r\n");
                size_t end = value.find_last_not_of(" \t\r\n");
                
                if (start == std::string::npos) {
                    std::cerr << "Warning: Empty value at line " << line_number << ", column " << col << std::endl;
                    row.push_back(0.0);
                    continue;
                }
                
                value = value.substr(start, end - start + 1);
                
                try {
                    double val = std::stod(value);
                    row.push_back(val);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid value '" << value << "' at line " << line_number 
                              << ", column " << col << std::endl;
                    throw std::runtime_error("Failed to parse CSV file at line " + std::to_string(line_number));
                }
            }
            
            if (!row.empty()) {
                data.push_back(row);
            }
        }
    }
    
    while (std::getline(file, line)) {
        line_number++;
        
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        int col = 0;
        
        while (std::getline(ss, value, ',')) {
            col++;
            size_t start = value.find_first_not_of(" \t\r\n");
            size_t end = value.find_last_not_of(" \t\r\n");
            
            if (start == std::string::npos) {
                std::cerr << "Warning: Empty value at line " << line_number << ", column " << col << std::endl;
                row.push_back(0.0);
                continue;
            }
            
            value = value.substr(start, end - start + 1);
            
            try {
                double val = std::stod(value);
                row.push_back(val);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid value '" << value << "' at line " << line_number 
                          << ", column " << col << std::endl;
                throw std::runtime_error("Failed to parse CSV file at line " + std::to_string(line_number));
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    
    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }
    
    int n_samples = data.size();
    int n_features = data[0].size() - 1;
    
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i].size() != data[0].size()) {
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(i + 1 + (has_header ? 1 : 0)));
        }
    }
    
    Dataset dataset(n_samples, n_features);
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            dataset.X(i, j) = data[i][j];
        }
        dataset.y[i] = data[i][n_features];
    }
    
    std::cout << "Loaded " << n_samples << " samples with " << n_features << " features" << std::endl;
    
    return dataset;
}

// BV18: Client-side preprocessing (before encryption)
// All operations here are ALLOWED - done in plaintext
void DataLoader::preprocess_bv18(Dataset& train, Dataset& test) {
    int n_features = train.n_features;
    
    std::cout << "=== BV18 Client-side Preprocessing (Plaintext) ===" << std::endl;
    
    // Step 1: Min-Max scaling to [0, 1]
    std::vector<double> min_vals(n_features);
    std::vector<double> max_vals(n_features);
    
    for (int j = 0; j < n_features; j++) {
        min_vals[j] = train.X(0, j);
        max_vals[j] = train.X(0, j);
        
        for (int i = 1; i < train.n_samples; i++) {
            if (train.X(i, j) < min_vals[j]) min_vals[j] = train.X(i, j);
            if (train.X(i, j) > max_vals[j]) max_vals[j] = train.X(i, j);
        }
    }
    
    for (int j = 0; j < n_features; j++) {
        double range = max_vals[j] - min_vals[j];
        if (range < 1e-8) range = 1.0;
        
        for (int i = 0; i < train.n_samples; i++) {
            train.X(i, j) = (train.X(i, j) - min_vals[j]) / range;
        }
        
        for (int i = 0; i < test.n_samples; i++) {
            test.X(i, j) = (test.X(i, j) - min_vals[j]) / range;
        }
    }
    std::cout << "Applied Min-Max scaling [0, 1]" << std::endl;
    
    // Step 2: Convert labels {0, 1} → {-1, +1}
    for (int i = 0; i < train.n_samples; i++) {
        if (train.y[i] == 0.0) {
            train.y[i] = -1.0;
        } else {
            train.y[i] = 1.0;
        }
    }
    
    for (int i = 0; i < test.n_samples; i++) {
        if (test.y[i] == 0.0) {
            test.y[i] = -1.0;
        } else {
            test.y[i] = 1.0;
        }
    }
    std::cout << "Converted labels to {-1, +1}" << std::endl;
    
    // Step 3: Add bias column (constant 1.0)
    Matrix X_train_new(train.n_samples, train.n_features + 1);
    Matrix X_test_new(test.n_samples, test.n_features + 1);
    
    for (int i = 0; i < train.n_samples; i++) {
        X_train_new(i, 0) = 1.0;
        for (int j = 0; j < train.n_features; j++) {
            X_train_new(i, j + 1) = train.X(i, j);
        }
    }
    
    for (int i = 0; i < test.n_samples; i++) {
        X_test_new(i, 0) = 1.0;
        for (int j = 0; j < test.n_features; j++) {
            X_test_new(i, j + 1) = test.X(i, j);
        }
    }
    
    train.X = X_train_new;
    test.X = X_test_new;
    train.n_features += 1;
    test.n_features += 1;
    std::cout << "Added bias column" << std::endl;
    
    std::cout << "=== Preprocessing Complete ===" << std::endl;
}