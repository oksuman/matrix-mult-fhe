// matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>

class Matrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;
    
    Matrix(int r, int c);
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;
};

class Vector {
public:
    std::vector<double> data;
    int size;
    
    Vector(int n);
    double& operator[](int i);
    const double& operator[](int i) const;
    double norm() const;
};

#endif