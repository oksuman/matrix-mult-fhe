#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

// Utility function to find nearest power of 2 >= n
inline size_t nextPowerOf2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

struct LDADataset {
    // Raw data
    std::vector<std::vector<double>> samples;  // [sample_idx][feature_idx]
    std::vector<int> labels;                    // class labels (0, 1, ...)

    // Dimensions
    size_t numSamples;
    size_t numFeatures;
    size_t numClasses;

    // Padded dimensions (power of 2)
    size_t paddedFeatures;   // f̃
    size_t paddedSamples;    // s̃ for all samples
    std::vector<size_t> paddedSamplesPerClass;  // s̃_c for each class
    std::vector<size_t> samplesPerClass;        // actual samples per class

    // Normalization parameters (for inference)
    std::vector<double> featureMin;
    std::vector<double> featureMax;
};

struct EncodedData {
    std::vector<double> allSamples;                    // All samples encoded (length: s̃ * f̃)
    std::vector<std::vector<double>> classSamples;     // Per-class samples (each length: s̃_c * f̃)

    size_t paddedFeatures;
    size_t paddedSamples;
    std::vector<size_t> paddedSamplesPerClass;
};

class LDADataEncoder {
public:
    // Save dataset to CSV (for reproducibility)
    static void saveToCSV(const LDADataset& dataset, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Write header
        for (size_t j = 0; j < dataset.numFeatures; j++) {
            file << "f" << j << ",";
        }
        file << "label\n";

        // Write data
        for (size_t i = 0; i < dataset.numSamples; i++) {
            for (size_t j = 0; j < dataset.numFeatures; j++) {
                file << std::setprecision(10) << dataset.samples[i][j] << ",";
            }
            file << dataset.labels[i] << "\n";
        }
        file.close();
    }

    // Load dataset from CSV
    static void loadFromCSV(const std::string& filename, LDADataset& dataset) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Skip header
        std::string header;
        std::getline(file, header);

        dataset.samples.clear();
        dataset.labels.clear();

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> sample;

            while (std::getline(ss, cell, ',')) {
                sample.push_back(std::stod(cell));
            }

            if (sample.empty()) continue;

            int label = static_cast<int>(sample.back());
            sample.pop_back();

            dataset.samples.push_back(sample);
            dataset.labels.push_back(label);
        }
        file.close();

        dataset.numSamples = dataset.samples.size();
        dataset.numFeatures = dataset.samples[0].size();

        // Find number of classes
        int maxLabel = *std::max_element(dataset.labels.begin(), dataset.labels.end());
        dataset.numClasses = maxLabel + 1;

        dataset.samplesPerClass.resize(dataset.numClasses, 0);
        for (int label : dataset.labels) {
            dataset.samplesPerClass[label]++;
        }

        // Calculate padded dimensions
        dataset.paddedFeatures = nextPowerOf2(dataset.numFeatures);
        dataset.paddedSamples = nextPowerOf2(dataset.numSamples);

        dataset.paddedSamplesPerClass.resize(dataset.numClasses);
        for (size_t c = 0; c < dataset.numClasses; c++) {
            dataset.paddedSamplesPerClass[c] = nextPowerOf2(dataset.samplesPerClass[c]);
        }

        dataset.featureMin.resize(dataset.numFeatures);
        dataset.featureMax.resize(dataset.numFeatures);
    }

    // Check if file exists
    static bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }

    // Load or create train/test split (for reproducibility)
    static void loadOrCreateSplit(const std::string& rawDataPath,
                                  const std::string& trainPath,
                                  const std::string& testPath,
                                  LDADataset& trainSet,
                                  LDADataset& testSet,
                                  double trainRatio = 0.8,
                                  unsigned int seed = 42) {
        if (fileExists(trainPath) && fileExists(testPath)) {
            // Load existing split
            std::cout << "Loading existing train/test split..." << std::endl;
            loadFromCSV(trainPath, trainSet);
            loadFromCSV(testPath, testSet);
            std::cout << "  Train: " << trainPath << " (" << trainSet.numSamples << " samples)" << std::endl;
            std::cout << "  Test:  " << testPath << " (" << testSet.numSamples << " samples)" << std::endl;
        } else {
            // Create new split
            std::cout << "Creating new train/test split..." << std::endl;
            loadAndSplitCSV(rawDataPath, trainSet, testSet, trainRatio, seed);

            // Save for future use
            saveToCSV(trainSet, trainPath);
            saveToCSV(testSet, testPath);
            std::cout << "  Saved train: " << trainPath << std::endl;
            std::cout << "  Saved test:  " << testPath << std::endl;
        }
    }

    // Load CSV and split into train/test (80/20)
    static void loadAndSplitCSV(const std::string& filename,
                                LDADataset& trainSet,
                                LDADataset& testSet,
                                double trainRatio = 0.8,
                                unsigned int seed = 42) {
        std::vector<std::vector<double>> allSamples;
        std::vector<int> allLabels;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Skip header (handle BOM if present)
        std::string header;
        std::getline(file, header);

        // Remove BOM if present
        if (header.size() >= 3 &&
            (unsigned char)header[0] == 0xEF &&
            (unsigned char)header[1] == 0xBB &&
            (unsigned char)header[2] == 0xBF) {
            header = header.substr(3);
        }

        // Read all data
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> sample;

            // Read all values
            std::vector<double> values;
            while (std::getline(ss, cell, ',')) {
                values.push_back(std::stod(cell));
            }

            if (values.empty()) continue;

            // Last column is the label
            int label = static_cast<int>(values.back());
            values.pop_back();

            allSamples.push_back(values);
            allLabels.push_back(label);
        }
        file.close();

        size_t totalSamples = allSamples.size();
        size_t numFeatures = allSamples[0].size();

        // Shuffle indices
        std::vector<size_t> indices(totalSamples);
        for (size_t i = 0; i < totalSamples; i++) indices[i] = i;

        std::mt19937 gen(seed);
        std::shuffle(indices.begin(), indices.end(), gen);

        // Split into train/test
        size_t trainSize = static_cast<size_t>(totalSamples * trainRatio);

        trainSet.numFeatures = numFeatures;
        testSet.numFeatures = numFeatures;

        for (size_t i = 0; i < trainSize; i++) {
            trainSet.samples.push_back(allSamples[indices[i]]);
            trainSet.labels.push_back(allLabels[indices[i]]);
        }

        for (size_t i = trainSize; i < totalSamples; i++) {
            testSet.samples.push_back(allSamples[indices[i]]);
            testSet.labels.push_back(allLabels[indices[i]]);
        }

        trainSet.numSamples = trainSet.samples.size();
        testSet.numSamples = testSet.samples.size();

        // Find number of classes and samples per class
        int maxLabel = *std::max_element(allLabels.begin(), allLabels.end());
        trainSet.numClasses = maxLabel + 1;
        testSet.numClasses = maxLabel + 1;

        trainSet.samplesPerClass.resize(trainSet.numClasses, 0);
        for (int label : trainSet.labels) {
            trainSet.samplesPerClass[label]++;
        }

        testSet.samplesPerClass.resize(testSet.numClasses, 0);
        for (int label : testSet.labels) {
            testSet.samplesPerClass[label]++;
        }

        // Calculate padded dimensions
        trainSet.paddedFeatures = nextPowerOf2(numFeatures);
        trainSet.paddedSamples = nextPowerOf2(trainSet.numSamples);

        trainSet.paddedSamplesPerClass.resize(trainSet.numClasses);
        for (size_t c = 0; c < trainSet.numClasses; c++) {
            trainSet.paddedSamplesPerClass[c] = nextPowerOf2(trainSet.samplesPerClass[c]);
        }

        testSet.paddedFeatures = trainSet.paddedFeatures;
        testSet.paddedSamples = nextPowerOf2(testSet.numSamples);

        testSet.paddedSamplesPerClass.resize(testSet.numClasses);
        for (size_t c = 0; c < testSet.numClasses; c++) {
            testSet.paddedSamplesPerClass[c] = nextPowerOf2(testSet.samplesPerClass[c]);
        }

        // Initialize normalization params
        trainSet.featureMin.resize(numFeatures);
        trainSet.featureMax.resize(numFeatures);
        testSet.featureMin.resize(numFeatures);
        testSet.featureMax.resize(numFeatures);
    }

    // Normalize features to [-1, 1] range per feature
    static void normalizeFeatures(LDADataset& dataset) {
        size_t numFeatures = dataset.numFeatures;
        size_t numSamples = dataset.numSamples;

        // Find min/max for each feature
        for (size_t f = 0; f < numFeatures; f++) {
            double minVal = dataset.samples[0][f];
            double maxVal = dataset.samples[0][f];

            for (size_t s = 1; s < numSamples; s++) {
                minVal = std::min(minVal, dataset.samples[s][f]);
                maxVal = std::max(maxVal, dataset.samples[s][f]);
            }

            dataset.featureMin[f] = minVal;
            dataset.featureMax[f] = maxVal;

            // Normalize to [-1, 1]
            double range = maxVal - minVal;
            if (range > 1e-10) {
                for (size_t s = 0; s < numSamples; s++) {
                    dataset.samples[s][f] = 2.0 * (dataset.samples[s][f] - minVal) / range - 1.0;
                }
            } else {
                // Constant feature, set to 0
                for (size_t s = 0; s < numSamples; s++) {
                    dataset.samples[s][f] = 0.0;
                }
            }
        }
    }

    // Normalize test set using train set's min/max
    static void normalizeWithParams(LDADataset& testSet, const LDADataset& trainSet) {
        size_t numFeatures = testSet.numFeatures;
        size_t numSamples = testSet.numSamples;

        testSet.featureMin = trainSet.featureMin;
        testSet.featureMax = trainSet.featureMax;

        for (size_t f = 0; f < numFeatures; f++) {
            double minVal = trainSet.featureMin[f];
            double maxVal = trainSet.featureMax[f];
            double range = maxVal - minVal;

            if (range > 1e-10) {
                for (size_t s = 0; s < numSamples; s++) {
                    testSet.samples[s][f] = 2.0 * (testSet.samples[s][f] - minVal) / range - 1.0;
                }
            } else {
                for (size_t s = 0; s < numSamples; s++) {
                    testSet.samples[s][f] = 0.0;
                }
            }
        }
    }

    // Encode dataset for plaintext computation
    // Uses actual padded dimensions (s_tilde × f_tilde)
    static EncodedData encodePlaintext(const LDADataset& dataset) {
        EncodedData encoded;

        size_t f_tilde = dataset.paddedFeatures;
        size_t s_tilde = dataset.paddedSamples;
        size_t numClasses = dataset.numClasses;

        encoded.paddedFeatures = f_tilde;
        encoded.paddedSamples = s_tilde;
        encoded.paddedSamplesPerClass = dataset.paddedSamplesPerClass;

        // Encode all samples: length = s̃ * f̃
        encoded.allSamples.resize(s_tilde * f_tilde, 0.0);

        for (size_t i = 0; i < dataset.numSamples; i++) {
            for (size_t j = 0; j < dataset.numFeatures; j++) {
                encoded.allSamples[i * f_tilde + j] = dataset.samples[i][j];
            }
        }

        // Encode per-class samples
        encoded.classSamples.resize(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            size_t s_tilde_c = dataset.paddedSamplesPerClass[c];
            encoded.classSamples[c].resize(s_tilde_c * f_tilde, 0.0);

            size_t classIdx = 0;
            for (size_t i = 0; i < dataset.numSamples; i++) {
                if (dataset.labels[i] == static_cast<int>(c)) {
                    for (size_t j = 0; j < dataset.numFeatures; j++) {
                        encoded.classSamples[c][classIdx * f_tilde + j] = dataset.samples[i][j];
                    }
                    classIdx++;
                }
            }
        }

        return encoded;
    }

    // Encode dataset into CKKS-friendly vectors for encrypted computation
    // For JKLS18 256×256 matrix multiplication:
    // Each sample occupies largeDim (256) slots per row
    // Total vector length: largeDim * largeDim = 65536
    static EncodedData encode(const LDADataset& dataset, size_t largeDim = 256) {
        EncodedData encoded;

        size_t f_tilde = dataset.paddedFeatures;
        size_t s_tilde = dataset.paddedSamples;
        size_t numClasses = dataset.numClasses;

        encoded.paddedFeatures = f_tilde;
        encoded.paddedSamples = s_tilde;
        encoded.paddedSamplesPerClass = dataset.paddedSamplesPerClass;

        // Encode all samples as largeDim × largeDim matrix
        // Each row has largeDim slots (features padded with zeros)
        encoded.allSamples.resize(largeDim * largeDim, 0.0);

        for (size_t i = 0; i < dataset.numSamples; i++) {
            for (size_t j = 0; j < dataset.numFeatures; j++) {
                encoded.allSamples[i * largeDim + j] = dataset.samples[i][j];
            }
        }

        // Encode per-class samples as largeDim × largeDim matrices
        encoded.classSamples.resize(numClasses);

        for (size_t c = 0; c < numClasses; c++) {
            encoded.classSamples[c].resize(largeDim * largeDim, 0.0);

            size_t classIdx = 0;
            for (size_t i = 0; i < dataset.numSamples; i++) {
                if (dataset.labels[i] == static_cast<int>(c)) {
                    for (size_t j = 0; j < dataset.numFeatures; j++) {
                        encoded.classSamples[c][classIdx * largeDim + j] = dataset.samples[i][j];
                    }
                    classIdx++;
                }
            }
        }

        return encoded;
    }

    // Debug print
    static void printDatasetInfo(const LDADataset& dataset, const std::string& name) {
        std::cout << "=== " << name << " ===" << std::endl;
        std::cout << "Samples: " << dataset.numSamples << " (padded: " << dataset.paddedSamples << ")" << std::endl;
        std::cout << "Features: " << dataset.numFeatures << " (padded: " << dataset.paddedFeatures << ")" << std::endl;
        std::cout << "Classes: " << dataset.numClasses << std::endl;
        for (size_t c = 0; c < dataset.numClasses; c++) {
            std::cout << "  Class " << c << ": " << dataset.samplesPerClass[c]
                      << " samples (padded: " << dataset.paddedSamplesPerClass[c] << ")" << std::endl;
        }
        std::cout << std::endl;
    }
};
