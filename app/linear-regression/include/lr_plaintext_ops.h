// lr_plaintext_ops.h - Plaintext matrix operations for Linear Regression
#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

class LRPlaintextOps {
public:
    // Matrix multiplication: C = A * B (d x d matrices in row-major)
    static std::vector<double> matMult(const std::vector<double>& A,
                                       const std::vector<double>& B, int d) {
        std::vector<double> C(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                for (int k = 0; k < d; k++) {
                    C[i * d + j] += A[i * d + k] * B[k * d + j];
                }
            }
        }
        return C;
    }

    // Matrix transpose
    static std::vector<double> transpose(const std::vector<double>& A, int d) {
        std::vector<double> At(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                At[j * d + i] = A[i * d + j];
            }
        }
        return At;
    }

    // Matrix trace
    static double trace(const std::vector<double>& A, int d) {
        double tr = 0.0;
        for (int i = 0; i < d; i++) {
            tr += A[i * d + i];
        }
        return tr;
    }

    // Scalar multiplication
    static std::vector<double> multScalar(const std::vector<double>& A, double s) {
        std::vector<double> result(A.size());
        for (size_t i = 0; i < A.size(); i++) {
            result[i] = A[i] * s;
        }
        return result;
    }

    // Element-wise subtraction
    static std::vector<double> sub(const std::vector<double>& A, const std::vector<double>& B) {
        std::vector<double> result(A.size());
        for (size_t i = 0; i < A.size(); i++) {
            result[i] = A[i] - B[i];
        }
        return result;
    }

    // Element-wise addition
    static std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B) {
        std::vector<double> result(A.size());
        for (size_t i = 0; i < A.size(); i++) {
            result[i] = A[i] + B[i];
        }
        return result;
    }

    // Identity matrix
    static std::vector<double> identity(int d) {
        std::vector<double> I(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            I[i * d + i] = 1.0;
        }
        return I;
    }

    // Matrix inversion using iterative method
    static std::vector<double> invertMatrix(const std::vector<double>& A, int d,
                                            int iterations, bool verbose = false) {
        auto I = identity(d);
        double tr = trace(A, d);

        if (verbose) {
            std::cout << "  [Inversion] trace(A) = " << tr << std::endl;
        }

        if (std::abs(tr) < 1e-10) {
            std::cerr << "Matrix appears singular (trace ~ 0)" << std::endl;
            return I;
        }

        // Y_0 = (1/trace) * I
        double alpha = 1.0 / tr;
        auto Y = multScalar(I, alpha);

        // A_bar = I - alpha * A
        auto A_bar = sub(I, multScalar(A, alpha));

        if (verbose) {
            std::cout << "  [Inversion] alpha = " << alpha << std::endl;
            std::cout << "  [Inversion] A_bar[0,0] = " << A_bar[0] << std::endl;
        }

        for (int iter = 0; iter < iterations - 1; iter++) {
            // Y = Y * (I + A_bar)
            auto I_plus_Abar = add(I, A_bar);
            Y = matMult(Y, I_plus_Abar, d);

            // A_bar = A_bar * A_bar
            A_bar = matMult(A_bar, A_bar, d);

            if (verbose && (iter < 3 || iter == iterations - 2)) {
                std::cout << "  [Iter " << iter << "] Y[0,0]=" << Y[0]
                          << " A_bar[0,0]=" << A_bar[0] << std::endl;
            }
        }

        // Final iteration
        auto I_plus_Abar = add(I, A_bar);
        Y = matMult(Y, I_plus_Abar, d);

        return Y;
    }

    // Matrix-vector multiplication (for X^T * y)
    static std::vector<double> matVecMult(const std::vector<double>& A,
                                          const std::vector<double>& v, int rows, int cols) {
        std::vector<double> result(rows, 0.0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += A[i * cols + j] * v[j];
            }
        }
        return result;
    }

    // Rebatch: Extract f×f from d×d matrix (top-left corner)
    static std::vector<double> rebatch(const std::vector<double>& A, int d, int f) {
        std::vector<double> result(f * f, 0.0);
        for (int i = 0; i < f; i++) {
            for (int j = 0; j < f; j++) {
                result[i * f + j] = A[i * d + j];
            }
        }
        return result;
    }

    // Print matrix
    static void printMatrix(const std::string& label, const std::vector<double>& A,
                           int rows, int cols, int maxShow = 5) {
        std::cout << "=== " << label << " (" << rows << "x" << cols << ") ===" << std::endl;
        int showRows = std::min(rows, maxShow);
        int showCols = std::min(cols, maxShow);
        for (int i = 0; i < showRows; i++) {
            std::cout << "  ";
            for (int j = 0; j < showCols; j++) {
                std::cout << std::setw(12) << std::setprecision(6) << std::fixed << A[i * cols + j];
            }
            if (cols > maxShow) std::cout << " ...";
            std::cout << std::endl;
        }
        if (rows > maxShow) std::cout << "  ..." << std::endl;
        std::cout << std::endl;
    }

    // Print vector
    static void printVector(const std::string& label, const std::vector<double>& v,
                           int maxShow = 8) {
        std::cout << "=== " << label << " (len=" << v.size() << ") ===" << std::endl;
        std::cout << "  ";
        int show = std::min((int)v.size(), maxShow);
        for (int i = 0; i < show; i++) {
            std::cout << std::setw(12) << std::setprecision(6) << std::fixed << v[i];
        }
        if ((int)v.size() > maxShow) std::cout << " ...";
        std::cout << std::endl << std::endl;
    }

    static std::vector<double> simulateEncryptedMatVecMult(
        const std::vector<double>& inv_XtX,
        const std::vector<double>& Xty,
        int d, bool verbose = true) {

        std::cout << "\n=== Simulating Encrypted Mat-Vec Mult ===" << std::endl;

        std::vector<double> Xty_transposed(d * d);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                Xty_transposed[i * d + j] = Xty[i];
            }
        }

        if (verbose) {
            std::cout << "Xty_transposed:" << std::endl;
            for (int i = 0; i < d; i++) {
                std::cout << "  row " << i << ": ";
                for (int j = 0; j < d; j++) {
                    std::cout << std::setw(10) << std::setprecision(4) << Xty_transposed[i * d + j];
                }
                std::cout << std::endl;
            }
        }

        std::vector<double> result(d * d);
        for (int i = 0; i < d * d; i++) {
            result[i] = inv_XtX[i] * Xty_transposed[i];
        }

        if (verbose) {
            std::cout << "\nElement-wise mult result:" << std::endl;
            for (int i = 0; i < d; i++) {
                std::cout << "  row " << i << ": ";
                for (int j = 0; j < d; j++) {
                    std::cout << std::setw(10) << std::setprecision(2) << result[i * d + j];
                }
                std::cout << std::endl;
            }
        }

        std::cout << "\nFolding sum (32, 16, 8):" << std::endl;
        for (int shift = d * d / 2; shift >= d; shift /= 2) {
            std::cout << "  rotate by " << shift << ":" << std::endl;
            std::vector<double> rotated(d * d);
            for (int i = 0; i < d * d; i++) {
                rotated[i] = result[(i + shift) % (d * d)];
            }
            for (int i = 0; i < d * d; i++) {
                result[i] += rotated[i];
            }
            for (int row = 0; row < d; row++) {
                std::cout << "    row " << row << ": ";
                for (int j = 0; j < d; j++) {
                    std::cout << std::setw(10) << std::setprecision(4) << result[row * d + j];
                }
                std::cout << std::endl;
            }
        }

        std::cout << "\nFinal result (all rows):" << std::endl;
        for (int i = 0; i < d; i++) {
            std::cout << "  row " << i << ": ";
            for (int j = 0; j < d; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << result[i * d + j];
            }
            std::cout << std::endl;
        }

        return result;
    }
};
