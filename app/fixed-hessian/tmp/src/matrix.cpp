// matrix.cpp
#include "matrix.h"

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(r, std::vector<double>(c, 0.0));
}

double& Matrix::operator()(int i, int j) { 
    return data[i][j]; 
}

const double& Matrix::operator()(int i, int j) const { 
    return data[i][j]; 
}

Vector::Vector(int n) : size(n), data(n, 0.0) {}

double& Vector::operator[](int i) { 
    return data[i]; 
}

const double& Vector::operator[](int i) const { 
    return data[i]; 
}

double Vector::norm() const {
    double sum = 0.0;
    for (double v : data) sum += v * v;
    return std::sqrt(sum);
}