#include "logistic_regression.h"
#include "data_loader.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

// BV18: labels are {-1, +1}, predictions are [0, 1]
double calculate_accuracy_bv18(const Vector& y_true, const Vector& y_pred) {
    int correct = 0;
    for (int i = 0; i < y_true.size; i++) {
        int pred_class = (y_pred[i] >= 0.5) ? 1 : -1;
        if (pred_class == static_cast<int>(y_true[i])) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_true.size;
}

void save_model(const Vector& weights, const std::string& model_name, const std::string& method) {
    mkdir("../model", 0755);
    
    std::string filename = "../model/" + model_name + "_" + method + ".txt";
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create model file: " + filename);
    }
    
    file << std::fixed << std::setprecision(10);
    file << "bias: " << weights[0] << std::endl;
    file << "weights:" << std::endl;
    for (int i = 1; i < weights.size; i++) {
        file << weights[i] << std::endl;
    }
    
    file.close();
    std::cout << "Model saved to: " << filename << std::endl;
}

Vector load_model(const std::string& model_path) {
    std::ifstream file(model_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }
    
    std::vector<double> weights_vec;
    std::string line;
    
    std::getline(file, line);
    size_t pos = line.find(": ");
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid model file format");
    }
    double bias = std::stod(line.substr(pos + 2));
    weights_vec.push_back(bias);
    
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            weights_vec.push_back(std::stod(line));
        }
    }
    
    file.close();
    
    Vector weights(weights_vec.size());
    for (size_t i = 0; i < weights_vec.size(); i++) {
        weights[i] = weights_vec[i];
    }
    
    return weights;
}

void print_help(const char* program_name) {
    std::cout << "BV18 Logistic Regression - HE-Friendly Implementation" << std::endl;
    std::cout << "======================================================" << std::endl << std::endl;
    
    std::cout << "Usage:" << std::endl;
    std::cout << "  Training:  " << program_name << " -train <train_file> -test <test_file> [-h|-fh|-fhd] [-iter <num>] [-m <model_name>]" << std::endl;
    std::cout << "  Testing:   " << program_name << " -test <test_file> -m <model_path>" << std::endl;
    std::cout << "  Help:      " << program_name << " -help" << std::endl << std::endl;
    
    std::cout << "BV18 Training Methods:" << std::endl;
    std::cout << "  -h     Full Hessian (HE: expensive, many levels)" << std::endl;
    std::cout << "  -fh    Fixed Hessian (HE: moderate cost)" << std::endl;
    std::cout << "  -fhd   Simplified Fixed Hessian - SFH (HE: MOST EFFICIENT) ★" << std::endl;
    std::cout << "  Note: You must specify exactly ONE method" << std::endl << std::endl;
    
    std::cout << "Options:" << std::endl;
    std::cout << "  -train <file>      Path to training CSV file" << std::endl;
    std::cout << "  -test <file>       Path to test CSV file" << std::endl;
    std::cout << "  -iter <num>        Number of iterations (default: 100, FIXED)" << std::endl;
    std::cout << "  -b <batch_size>    Batch size for processing (default: full-batch)" << std::endl;
    std::cout << "  -m <name/path>     Model name (train) or path (test)" << std::endl;
    std::cout << "  -help              Display this help" << std::endl << std::endl;
    
    std::cout << "BV18 HE-Friendly Design:" << std::endl;
    std::cout << "  Client-side (plaintext):" << std::endl;
    std::cout << "    - Min-Max scaling to [0, 1]" << std::endl;
    std::cout << "    - Label conversion {0, 1} → {-1, +1}" << std::endl;
    std::cout << "    - Bias column addition" << std::endl;
    std::cout << "    - Scale by 1/2 (level optimization)" << std::endl;
    std::cout << "  Server-side (encrypted):" << std::endl;
    std::cout << "    - Sigmoid approximation: σ(x) ≈ 1/2 + x/4" << std::endl;
    std::cout << "    - Fixed iterations (no convergence check)" << std::endl;
    std::cout << "    - Polynomial operations only (+, -, *)" << std::endl;
    std::cout << "    - Newton-Raphson inverse (for SFH)" << std::endl << std::endl;
    
    std::cout << "Examples:" << std::endl;
    std::cout << "  # SFH method (recommended for HE)" << std::endl;
    std::cout << "  " << program_name << " -train train.csv -test test.csv -fhd -iter 100" << std::endl << std::endl;
    
    std::cout << "  # SFH with batch processing (HE simulation)" << std::endl;
    std::cout << "  " << program_name << " -train train.csv -test test.csv -fhd -iter 100 -b 128" << std::endl << std::endl;
    
    std::cout << "  # Fixed Hessian" << std::endl;
    std::cout << "  " << program_name << " -train train.csv -test test.csv -fh -iter 50" << std::endl << std::endl;
    
    std::cout << "  # Save model" << std::endl;
    std::cout << "  " << program_name << " -train train.csv -test test.csv -fhd -m diabetes" << std::endl << std::endl;
    
    std::cout << "  # Test with saved model" << std::endl;
    std::cout << "  " << program_name << " -test test.csv -m ../model/diabetes_sfh.txt" << std::endl << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  Training:  " << program_name << " -train <train_file> -test <test_file> [-h|-fh|-fhd] [-iter <num>] [-m <model_name>]" << std::endl;
    std::cout << "  Testing:   " << program_name << " -test <test_file> -m <model_path>" << std::endl;
    std::cout << "  Help:      " << program_name << " -help" << std::endl;
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
    }
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string train_file;
    std::string test_file;
    std::string model_arg;
    bool has_model_option = false;
    bool use_hessian = false;
    bool use_fixed_hessian = false;
    bool use_fixed_hessian_diagonal = false;
    int method_count = 0;
    int max_iterations = 100;
    int batch_size = -1;
    bool has_batch_option = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-train") == 0 && i + 1 < argc) {
            train_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-test") == 0 && i + 1 < argc) {
            test_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_arg = argv[i + 1];
            has_model_option = true;
            i++;
        } else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc) {
            max_iterations = std::stoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            batch_size = std::stoi(argv[i + 1]);
            has_batch_option = true;
            if (batch_size <= 0) {
                std::cerr << "Error: Batch size must be positive" << std::endl;
                return 1;
            }
            i++;
        } else if (strcmp(argv[i], "-h") == 0) {
            use_hessian = true;
            method_count++;
        } else if (strcmp(argv[i], "-fh") == 0) {
            use_fixed_hessian = true;
            method_count++;
        } else if (strcmp(argv[i], "-fhd") == 0) {
            use_fixed_hessian_diagonal = true;
            method_count++;
        } else {
            std::cerr << "Error: Unknown option '" << argv[i] << "'" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (test_file.empty()) {
        std::cerr << "Error: Test file is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (!train_file.empty()) {
        if (method_count == 0) {
            std::cerr << "Error: You must specify exactly ONE training method (-h, -fh, or -fhd)" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        if (method_count > 1) {
            std::cerr << "Error: You can only specify ONE training method at a time" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << std::fixed << std::setprecision(4);
    
    try {
        if (!train_file.empty()) {
            // TRAINING MODE
            std::cout << "=== BV18 TRAINING MODE ===" << std::endl;
            std::cout << "Train file: " << train_file << std::endl;
            std::cout << "Test file: " << test_file << std::endl;
            
            if (has_model_option) {
                std::cout << "Model name: " << model_arg << std::endl;
            }
            
            std::cout << "Method: ";
            if (use_hessian) std::cout << "Full Hessian";
            if (use_fixed_hessian) std::cout << "Fixed Hessian";
            if (use_fixed_hessian_diagonal) std::cout << "Simplified Fixed Hessian (SFH)";
            std::cout << std::endl;
            
            std::cout << "Iterations: " << max_iterations << " (FIXED)" << std::endl;
            
            if (has_batch_option) {
                std::cout << "Batch size: " << batch_size << std::endl;
            } else {
                std::cout << "Batch mode: Full-batch" << std::endl;
            }
            std::cout << std::endl;
            
            Dataset train_data = DataLoader::load_csv(train_file, false);
            Dataset test_data = DataLoader::load_csv(test_file, false);
            
            std::cout << "Train samples: " << train_data.n_samples << ", Features: " << train_data.n_features << std::endl;
            std::cout << "Test samples: " << test_data.n_samples << ", Features: " << test_data.n_features << std::endl;
            
            if (has_batch_option && batch_size >= train_data.n_samples) {
                std::cout << "Warning: Batch size (" << batch_size << ") >= training samples (" 
                          << train_data.n_samples << "), using full batch instead" << std::endl;
                batch_size = -1;
            }
            
            std::cout << std::endl;
            
            // BV18 preprocessing (client-side, before encryption)
            DataLoader::preprocess_bv18(train_data, test_data);
            std::cout << std::endl;
            
            LogisticRegression lr(train_data.n_features);
            std::string method_name;
            
            if (use_hessian) {
                lr.fit_hessian_bv18(train_data.X, train_data.y, max_iterations, 0, batch_size);
                method_name = "full_hessian";
            } else if (use_fixed_hessian) {
                lr.fit_fixed_hessian_bv18(train_data.X, train_data.y, max_iterations, 0, batch_size);
                method_name = "fixed_hessian";
            } else if (use_fixed_hessian_diagonal) {
                lr.fit_fixed_hessian_diagonal_bv18(train_data.X, train_data.y, max_iterations, 0, batch_size);
                method_name = "sfh";
            }
            
            Vector pred_train = lr.predict_proba_bv18(train_data.X);
            Vector pred_test = lr.predict_proba_bv18(test_data.X);
            
            std::cout << std::endl;
            std::cout << "Train Accuracy: " << calculate_accuracy_bv18(train_data.y, pred_train) << std::endl;
            std::cout << "Test Accuracy:  " << calculate_accuracy_bv18(test_data.y, pred_test) << std::endl;
            
            if (has_model_option) {
                save_model(lr.get_weights(), model_arg, method_name);
            }
            
        } else {
            // TEST ONLY MODE
            if (!has_model_option) {
                std::cerr << "Error: -m option is required for test-only mode" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
            
            std::cout << "=== BV18 TEST MODE ===" << std::endl;
            std::cout << "Test file: " << test_file << std::endl;
            std::cout << "Model file: " << model_arg << std::endl;
            std::cout << std::endl;
            
            Vector weights = load_model(model_arg);
            std::cout << "Model loaded (weights: " << weights.size << ")" << std::endl;
            
            Dataset test_data = DataLoader::load_csv(test_file, false);
            Dataset dummy_train = DataLoader::load_csv(test_file, false);
            
            DataLoader::preprocess_bv18(dummy_train, test_data);
            
            if (test_data.n_features != weights.size) {
                std::cerr << "Error: Feature mismatch" << std::endl;
                return 1;
            }
            
            LogisticRegression lr(weights.size);
            for (int i = 0; i < weights.size; i++) {
                lr.get_weights()[i] = weights[i];
            }
            
            Vector pred_test = lr.predict_proba_bv18(test_data.X);
            std::cout << "Test Accuracy: " << calculate_accuracy_bv18(test_data.y, pred_test) << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}