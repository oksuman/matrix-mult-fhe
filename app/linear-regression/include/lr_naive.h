// lr_naive.h
// Naive (entry-level) linear regression using CKKS
// Each matrix element is a separate ciphertext (batchSize=1)
// Independent CryptoContext: no rotation keys, no bootstrapping
#pragma once

#include <openfhe.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace lbcrypto;

class LinearRegression_Naive {
public:
    static const int D = 8;   // FEATURE_DIM
    static const int N = 64;  // SAMPLE_DIM
    static const int MULT_DEPTH = 28;
    static const int INV_ITER = 18;
    static const int SCALAR_INV_ITER = 1;

private:
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    bool m_verbose;

    // Encrypted matrices
    std::vector<Ciphertext<DCRTPoly>> m_XtX;     // D*D ciphertexts
    std::vector<Ciphertext<DCRTPoly>> m_Xty;     // D ciphertexts
    std::vector<Ciphertext<DCRTPoly>> m_inv;      // D*D ciphertexts
    std::vector<Ciphertext<DCRTPoly>> m_weights;  // D ciphertexts

    Ciphertext<DCRTPoly> encryptScalar(double val) {
        std::vector<double> v(1, val);
        return m_cc->Encrypt(m_keyPair.publicKey,
            m_cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, 1));
    }

    // Entry-level d x d matrix multiply: C[i][j] = sum_k A[i][k]*B[k][j]
    std::vector<Ciphertext<DCRTPoly>> matMul(
        const std::vector<Ciphertext<DCRTPoly>>& A,
        const std::vector<Ciphertext<DCRTPoly>>& B, int d) {

        std::vector<Ciphertext<DCRTPoly>> C(d * d);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                auto sum = m_cc->EvalMultAndRelinearize(A[i * d], B[j]);
                for (int k = 1; k < d; k++) {
                    m_cc->EvalAddInPlace(sum,
                        m_cc->EvalMultAndRelinearize(A[i * d + k], B[k * d + j]));
                }
                C[i * d + j] = sum;
            }
        }
        return C;
    }

    // Entry-level matrix-vector multiply: w[j] = sum_k M[j][k]*v[k]
    std::vector<Ciphertext<DCRTPoly>> matVecMul(
        const std::vector<Ciphertext<DCRTPoly>>& M,
        const std::vector<Ciphertext<DCRTPoly>>& v, int d) {

        std::vector<Ciphertext<DCRTPoly>> w(d);
        for (int j = 0; j < d; j++) {
            w[j] = m_cc->EvalMultAndRelinearize(M[j * d], v[0]);
            for (int k = 1; k < d; k++) {
                m_cc->EvalAddInPlace(w[j],
                    m_cc->EvalMultAndRelinearize(M[j * d + k], v[k]));
            }
        }
        return w;
    }

    // Trace: sum of diagonal elements
    Ciphertext<DCRTPoly> evalTrace(
        const std::vector<Ciphertext<DCRTPoly>>& M, int d) {
        auto trace = M[0]->Clone();
        for (int i = 1; i < d; i++) {
            m_cc->EvalAddInPlace(trace, M[i * d + i]);
        }
        return trace;
    }

    // Scalar inverse via power series: 1/t
    Ciphertext<DCRTPoly> evalScalarInverse(
        const Ciphertext<DCRTPoly>& t, double upperBound, int iterations) {
        double x0 = 1.0 / upperBound;
        auto x = encryptScalar(x0);
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        for (int i = 0; i < iterations; i++) {
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }
        return x;
    }

    // Iterative matrix inversion: Y_0 = I/trace(M), A_bar_0 = I - M/trace(M)
    // Y = Y*(I+A_bar), A_bar = A_bar^2
    std::vector<Ciphertext<DCRTPoly>> evalInverse(
        const std::vector<Ciphertext<DCRTPoly>>& M, int d, int r) {

        auto trace = evalTrace(M, d);
        double traceUB = (double)N * d;
        auto alpha = evalScalarInverse(trace, traceUB, SCALAR_INV_ITER);

        if (m_verbose) {
            Plaintext ptxTrace;
            m_cc->Decrypt(m_keyPair.secretKey, trace, &ptxTrace);
            Plaintext ptxAlpha;
            m_cc->Decrypt(m_keyPair.secretKey, alpha, &ptxAlpha);
            std::cout << "  [Naive Inv] trace = " << ptxTrace->GetRealPackedValue()[0]
                      << ", alpha = " << ptxAlpha->GetRealPackedValue()[0] << std::endl;
        }

        // Y = alpha * I  (clone alpha for diagonal, encrypt 0 for off-diagonal)
        std::vector<Ciphertext<DCRTPoly>> Y(d * d);
        auto zeroAtAlphaLevel = m_cc->EvalSub(alpha, alpha);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i == j) {
                    Y[i * d + j] = alpha->Clone();
                } else {
                    Y[i * d + j] = zeroAtAlphaLevel->Clone();
                }
            }
        }

        // A_bar = I - alpha * M
        std::vector<Ciphertext<DCRTPoly>> A_bar(d * d);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                auto alphaM = m_cc->EvalMultAndRelinearize(alpha, M[i * d + j]);
                if (i == j) {
                    A_bar[i * d + j] = m_cc->EvalSub(1.0, alphaM);
                } else {
                    A_bar[i * d + j] = m_cc->EvalNegate(alphaM);
                }
            }
        }

        // r iterations: Y = Y*(I+A_bar), A_bar = A_bar^2
        for (int it = 0; it < r; it++) {
            // Build I + A_bar
            std::vector<Ciphertext<DCRTPoly>> IpA(d * d);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (i == j) {
                        IpA[i * d + j] = m_cc->EvalAdd(A_bar[i * d + j], 1.0);
                    } else {
                        IpA[i * d + j] = A_bar[i * d + j];
                    }
                }
            }

            Y = matMul(Y, IpA, d);

            if (it < r - 1) {
                A_bar = matMul(A_bar, A_bar, d);
            }

            if (m_verbose && (it % 5 == 0 || it == r - 1)) {
                std::cout << "  [Naive Iter " << it << "] Y[0][0] level: "
                          << Y[0]->GetLevel() << "/" << MULT_DEPTH << std::endl;
            }
        }

        return Y;
    }

public:
    LinearRegression_Naive() : m_verbose(false) {}

    void setVerbose(bool v) { m_verbose = v; }

    CryptoContext<DCRTPoly>& getCC() { return m_cc; }
    const KeyPair<DCRTPoly>& getKeyPair() const { return m_keyPair; }

    void setup() {
        CCParams<CryptoContextCKKSRNS> params;
        params.SetMultiplicativeDepth(MULT_DEPTH);
        params.SetScalingModSize(59);
        params.SetFirstModSize(60);
        params.SetBatchSize(1);
        params.SetSecurityLevel(HEStd_128_classic);

        m_cc = GenCryptoContext(params);
        m_cc->Enable(PKE);
        m_cc->Enable(KEYSWITCH);
        m_cc->Enable(LEVELEDSHE);

        m_keyPair = m_cc->KeyGen();
        m_cc->EvalMultKeyGen(m_keyPair.secretKey);

        if (m_verbose) {
            std::cout << "  [Naive Setup] multDepth=" << MULT_DEPTH
                      << ", batchSize=1"
                      << ", ringDim=" << m_cc->GetRingDimension()
                      << std::endl;
        }
    }

    // Row-by-row encryption and XtX/Xty accumulation
    // features: flat array, features[i * SAMPLE_DIM + j] = feature j of sample i
    // outcomes: outcomes[i] = label of sample i
    // Returns: (precomputation_time, total_CT_count)
    std::pair<std::chrono::duration<double>, int> computePrecomputation(
        const std::vector<double>& features,
        const std::vector<double>& outcomes) {

        using namespace std::chrono;
        auto start = high_resolution_clock::now();

        m_XtX.resize(D * D);
        m_Xty.resize(D);

        int totalCT = 0;

        // Process row 0: initialize accumulators
        {
            std::vector<Ciphertext<DCRTPoly>> x_row(D);
            for (int j = 0; j < D; j++) {
                x_row[j] = encryptScalar(features[0 * N + j]);
            }
            auto y_0 = encryptScalar(outcomes[0]);
            totalCT += D + 1;

            for (int j = 0; j < D; j++) {
                for (int k = 0; k < D; k++) {
                    m_XtX[j * D + k] = m_cc->EvalMultAndRelinearize(x_row[j], x_row[k]);
                }
                m_Xty[j] = m_cc->EvalMultAndRelinearize(x_row[j], y_0);
            }
        }

        // Process remaining rows: accumulate
        for (int i = 1; i < N; i++) {
            std::vector<Ciphertext<DCRTPoly>> x_row(D);
            for (int j = 0; j < D; j++) {
                x_row[j] = encryptScalar(features[i * N + j]);
            }
            auto y_i = encryptScalar(outcomes[i]);
            totalCT += D + 1;

            for (int j = 0; j < D; j++) {
                for (int k = 0; k < D; k++) {
                    m_cc->EvalAddInPlace(m_XtX[j * D + k],
                        m_cc->EvalMultAndRelinearize(x_row[j], x_row[k]));
                }
                m_cc->EvalAddInPlace(m_Xty[j],
                    m_cc->EvalMultAndRelinearize(x_row[j], y_i));
            }

            if (m_verbose && (i % 16 == 0)) {
                std::cout << "  [Row " << i << "/" << N
                          << "] XtX[0][0] level: " << m_XtX[0]->GetLevel()
                          << std::endl;
            }
        }

        if (m_verbose) {
            std::cout << "  [Precomp done] XtX level: " << m_XtX[0]->GetLevel()
                      << ", Xty level: " << m_Xty[0]->GetLevel()
                      << ", totalCT: " << totalCT << std::endl;

            // Verify XtX diagonal
            std::cout << "  [XtX diag] ";
            for (int i = 0; i < D; i++) {
                Plaintext ptx;
                m_cc->Decrypt(m_keyPair.secretKey, m_XtX[i * D + i], &ptx);
                std::cout << std::setprecision(4) << ptx->GetRealPackedValue()[0] << " ";
            }
            std::cout << std::endl;
        }

        auto end = high_resolution_clock::now();
        return {end - start, totalCT};
    }

    // Invert XtX
    std::chrono::duration<double> computeInverse() {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        m_inv = evalInverse(m_XtX, D, INV_ITER);
        auto end = high_resolution_clock::now();

        if (m_verbose) {
            std::cout << "  [Inv done] inv[0][0] level: " << m_inv[0]->GetLevel()
                      << "/" << MULT_DEPTH << std::endl;
            // Verify inverse diagonal
            std::cout << "  [inv diag] ";
            for (int i = 0; i < D; i++) {
                Plaintext ptx;
                m_cc->Decrypt(m_keyPair.secretKey, m_inv[i * D + i], &ptx);
                std::cout << std::setprecision(6) << ptx->GetRealPackedValue()[0] << " ";
            }
            std::cout << std::endl;
        }

        return end - start;
    }

    // Compute weights = inv(XtX) * Xty
    std::chrono::duration<double> computeWeights() {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        m_weights = matVecMul(m_inv, m_Xty, D);
        auto end = high_resolution_clock::now();

        if (m_verbose) {
            std::cout << "  [Weights] level: " << m_weights[0]->GetLevel()
                      << "/" << MULT_DEPTH << std::endl;
        }

        return end - start;
    }

    // Decrypt weights
    std::vector<double> getDecryptedWeights() {
        std::vector<double> weights(D);
        for (int j = 0; j < D; j++) {
            Plaintext ptx;
            m_cc->Decrypt(m_keyPair.secretKey, m_weights[j], &ptx);
            weights[j] = ptx->GetRealPackedValue()[0];
        }
        return weights;
    }

    // Get a sample ciphertext for size measurement
    Ciphertext<DCRTPoly> getSampleCiphertext() {
        if (!m_XtX.empty()) return m_XtX[0];
        return encryptScalar(0.0);
    }
};
