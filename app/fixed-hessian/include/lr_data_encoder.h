#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

struct LRDataset {
    std::vector<std::vector<double>> samples;  // [sample_idx][feature_idx]
    std::vector<int> labels;                    // {-1, +1}

    size_t numSamples;
    size_t numFeatures;

    // Min-max normalization parameters (from training set, for 13 raw features)
    std::vector<double> featureMin;
    std::vector<double> featureMax;
};

// Constants for HE packing - dataset dependent
#ifdef DATASET_DIABETES
    static const int LR_RAW_FEATURES = 8;         // original features (diabetes: 8)
    static const int LR_BATCH_SIZE = 64;          // samples per batch: 64
#else  // DATASET_HEART (default)
    static const int LR_RAW_FEATURES = 13;        // original features (heart disease: 13)
    static const int LR_BATCH_SIZE = 128;         // samples per batch: 128
#endif

static const int LR_FEATURES = 16;                // raw + 1 bias + padding to 16
static const int LR_MATRIX_DIM = LR_BATCH_SIZE;   // d: matrix dimension
static const int LR_SLOTS = LR_MATRIX_DIM * LR_MATRIX_DIM;  // d*d

// Unified iteration counts
static const int FH_SCALAR_INV_ITERATIONS = 2;

// Matrix inversion iterations by dimension (95th percentile)
inline int getFHInversionIterations(int d) {
    switch(d) {
        case 4:  return 18;
        case 8:  return 22;
        case 16: return 25;
        case 32: return 27;
        case 64: return 31;
        default: return 25;
    }
}

class LRDataEncoder {
public:
    // Load CSV (header: age,sex,...,thal,target)
    static LRDataset loadCSV(const std::string& filename) {
        LRDataset dataset;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Skip header
        std::string header;
        std::getline(file, header);

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> values;

            while (std::getline(ss, cell, ',')) {
                size_t start = cell.find_first_not_of(" \t\r\n");
                size_t end = cell.find_last_not_of(" \t\r\n");
                if (start != std::string::npos) {
                    cell = cell.substr(start, end - start + 1);
                }
                values.push_back(std::stod(cell));
            }

            if (values.empty()) continue;

            // Last column is label
            int rawLabel = static_cast<int>(values.back());
            values.pop_back();

            dataset.samples.push_back(values);
            // Convert {0} -> {-1}, {>0} -> {+1}
            dataset.labels.push_back(rawLabel == 0 ? -1 : 1);
        }
        file.close();

        dataset.numSamples = dataset.samples.size();
        dataset.numFeatures = dataset.samples[0].size();  // 13 at this point

        return dataset;
    }

    // Min-max normalization to [0,1] (compute min/max from this dataset, 13 raw features only)
    static void normalizeFeatures(LRDataset& dataset) {
        size_t f = dataset.numFeatures;
        size_t n = dataset.numSamples;

        dataset.featureMin.resize(f);
        dataset.featureMax.resize(f);

        for (size_t j = 0; j < f; j++) {
            double mn = dataset.samples[0][j];
            double mx = dataset.samples[0][j];
            for (size_t i = 1; i < n; i++) {
                if (dataset.samples[i][j] < mn) mn = dataset.samples[i][j];
                if (dataset.samples[i][j] > mx) mx = dataset.samples[i][j];
            }
            dataset.featureMin[j] = mn;
            dataset.featureMax[j] = mx;

            double range = mx - mn;
            if (range > 1e-10) {
                for (size_t i = 0; i < n; i++) {
                    dataset.samples[i][j] = (dataset.samples[i][j] - mn) / range;
                }
            } else {
                for (size_t i = 0; i < n; i++) {
                    dataset.samples[i][j] = 0.0;
                }
            }
        }
    }

    // Normalize test set using train set's min/max parameters
    static void normalizeWithParams(LRDataset& testSet, const LRDataset& trainSet) {
        size_t f = testSet.numFeatures;
        size_t n = testSet.numSamples;

        testSet.featureMin = trainSet.featureMin;
        testSet.featureMax = trainSet.featureMax;

        for (size_t j = 0; j < f; j++) {
            double mn = trainSet.featureMin[j];
            double mx = trainSet.featureMax[j];
            double range = mx - mn;

            if (range > 1e-10) {
                for (size_t i = 0; i < n; i++) {
                    testSet.samples[i][j] = (testSet.samples[i][j] - mn) / range;
                }
            } else {
                for (size_t i = 0; i < n; i++) {
                    testSet.samples[i][j] = 0.0;
                }
            }
        }
    }

    // Add bias column (constant 1.0) and pad to 16 features
    static void addBiasAndPad(LRDataset& dataset) {
        for (size_t i = 0; i < dataset.numSamples; i++) {
            // Add bias
            dataset.samples[i].push_back(1.0);
            // Pad remaining to reach 16 features
            while (dataset.samples[i].size() < LR_FEATURES) {
                dataset.samples[i].push_back(0.0);
            }
        }
        dataset.numFeatures = LR_FEATURES;  // 16
    }

    // Limit to maxSamples (truncate)
    static void limitSamples(LRDataset& dataset, size_t maxSamples) {
        if (dataset.numSamples <= maxSamples) return;
        dataset.samples.resize(maxSamples);
        dataset.labels.resize(maxSamples);
        dataset.numSamples = maxSamples;
    }

    // Pack batch b of X into 64x64 = 4096-element vector
    // Each row: 16 features + 48 zeros
    static std::vector<double> packBatchX(const LRDataset& dataset, int batchIdx) {
        std::vector<double> packed(LR_SLOTS, 0.0);
        int startSample = batchIdx * LR_BATCH_SIZE;

        for (int i = 0; i < LR_BATCH_SIZE; i++) {
            int sampleIdx = startSample + i;
            if (sampleIdx >= (int)dataset.numSamples) break;

            for (int j = 0; j < LR_FEATURES; j++) {
                packed[i * LR_MATRIX_DIM + j] = dataset.samples[sampleIdx][j];
            }
        }
        return packed;
    }

    // Pack batch b of y into 64-element vector
    static std::vector<double> packBatchY(const LRDataset& dataset, int batchIdx) {
        std::vector<double> packed(LR_BATCH_SIZE, 0.0);
        int startSample = batchIdx * LR_BATCH_SIZE;

        for (int i = 0; i < LR_BATCH_SIZE; i++) {
            int sampleIdx = startSample + i;
            if (sampleIdx >= (int)dataset.numSamples) break;
            packed[i] = dataset.labels[sampleIdx];
        }
        return packed;
    }

    static int getNumBatches(const LRDataset& dataset) {
        return (int)dataset.numSamples / LR_BATCH_SIZE;
    }

    static void printDatasetInfo(const LRDataset& dataset, const std::string& name) {
        std::cout << "=== " << name << " ===" << std::endl;
        std::cout << "Samples: " << dataset.numSamples << std::endl;
        std::cout << "Features: " << dataset.numFeatures << std::endl;

        int pos = 0, neg = 0;
        for (int l : dataset.labels) {
            if (l == 1) pos++;
            else neg++;
        }
        std::cout << "Labels: +" << pos << " / -" << neg << std::endl;
        std::cout << std::endl;
    }
};
