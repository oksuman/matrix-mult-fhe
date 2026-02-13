#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "matrix.h"
#include <string>

struct Dataset {
    Matrix X;
    Vector y;
    int n_samples;
    int n_features;
    
    Dataset(int samples, int features);
};

class DataLoader {
public:
    static Dataset load_csv(const std::string& filename, bool has_header = true);
    
    // BV18: All preprocessing done BEFORE encryption (client-side in plaintext)
    // Includes: Min-Max [0,1], label conversion {-1,+1}, bias addition, scale by 1/2
    static void preprocess_bv18(Dataset& train, Dataset& test);
};

#endif