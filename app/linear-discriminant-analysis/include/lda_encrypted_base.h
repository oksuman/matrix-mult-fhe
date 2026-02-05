// lda_encrypted_base.h
#pragma once

#include "encryption.h"
#include "rotation.h"
#include "lda_data_encoder.h"
#include <memory>
#include <chrono>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <set>

// HD Dataset constants
// f=13 features, s=242 samples
// Padded: f̃=16, s̃=256
// Matrix dimension for multiplication: max(s̃, f̃) = 256
const int HD_FEATURE_DIM = 13;      // Actual features
const int HD_SAMPLE_DIM = 242;      // Actual samples (train set ~80% of 303)
const int HD_PADDED_FEATURE = 16;   // Padded to power of 2
const int HD_PADDED_SAMPLE = 256;   // Padded to power of 2
const int HD_MATRIX_DIM = 256;      // max(s̃, f̃) for JKLS18 matrix mult

struct LDAEncryptedResult {
    Ciphertext<DCRTPoly> Sw_inv;         // S_W^{-1} (f̃ x f̃ = 16x16)
    Ciphertext<DCRTPoly> Sw_inv_Sb;      // S_W^{-1} * S_B (optional)
    std::vector<Ciphertext<DCRTPoly>> classMeansEncrypted;  // Encrypted class means
    Ciphertext<DCRTPoly> globalMeanEncrypted;               // Encrypted global mean

    // Decrypted values for inference (done on client side)
    std::vector<std::vector<double>> classMeans;
    std::vector<double> globalMean;
    std::vector<double> Sw_decrypted;     // S_W matrix (for debugging)
    std::vector<double> Sb_decrypted;     // S_B matrix (for debugging)
    std::vector<double> Sw_inv_decrypted;
    std::vector<size_t> classCounts;

    // Intermediate results for debugging
    std::vector<std::vector<double>> X_bar_c_decrypted;   // Centered data per class
    std::vector<std::vector<double>> S_c_decrypted;       // Scatter matrix per class (before rebatch)
};

struct LDATimingResult {
    std::chrono::duration<double> meanComputation;
    std::chrono::duration<double> swComputation;
    std::chrono::duration<double> sbComputation;
    std::chrono::duration<double> inversionTime;
    std::chrono::duration<double> totalTime;
};

class LDAEncryptedBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    std::vector<int> m_rotIndices;
    RotationComposer rot;
    int m_multDepth;
    bool m_useBootstrapping;

    // ============ Utility Functions ============

    std::vector<double> vectorRotate(const std::vector<double>& vec, int rotateIndex) {
        if (vec.empty()) return std::vector<double>();
        std::vector<double> result = vec;
        int n = result.size();
        rotateIndex = ((rotateIndex % n) + n) % n;
        if (rotateIndex > 0) {
            std::rotate(result.begin(), result.begin() + rotateIndex, result.end());
        }
        return result;
    }

    std::vector<double> initializeIdentityMatrix(size_t dim) {
        std::vector<double> identity(dim * dim, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

    std::vector<double> initializeIdentityMatrix2(size_t dim, int batchSize) {
        std::vector<double> identity(batchSize, 0.0);
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = 1.0;
        }
        return identity;
    }

    Ciphertext<DCRTPoly> getZeroCiphertext(int batchSize) {
        std::vector<double> zeroVec(batchSize, 0.0);
        auto zeroPtx = m_cc->MakeCKKSPackedPlaintext(zeroVec, 1, 0, nullptr, batchSize);
        return m_cc->Encrypt(zeroPtx, m_keyPair.publicKey);
    }

    // ============ Transpose Mask Generation ============

    std::vector<double> generateTransposeMask(int k, int d) {
        std::set<int> indices;
        if (k >= 0) {
            for (int j = 0; j < d - k; j++) {
                indices.insert((d + 1) * j + k);
            }
        } else {
            for (int j = -k; j < d; j++) {
                indices.insert((d + 1) * j + k);
            }
        }
        std::vector<double> msk(d * d, 0);
        for (int index : indices) {
            msk[index] = 1.0;
        }
        return msk;
    }

    // ============ Matrix Transpose ============

    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        auto p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(0, d), 1, 0, nullptr, batchSize);
        auto M_transposed = m_cc->EvalMult(M, p);

        for (int i = 1; i < d; i++) {
            p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(i, d), 1, 0, nullptr, batchSize);
            m_cc->EvalAddInPlace(M_transposed,
                                m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }

        for (int i = -1; i > -d; i--) {
            p = m_cc->MakeCKKSPackedPlaintext(generateTransposeMask(i, d), 1, 0, nullptr, batchSize);
            m_cc->EvalAddInPlace(M_transposed,
                                m_cc->EvalMult(rot.rotate(M, (d - 1) * i), p));
        }
        return M_transposed;
    }

    // ============ Matrix Trace ============

    Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        std::vector<double> msk(batchSize, 0);
        for (int i = 0; i < d * d; i += (d + 1)) {
            msk[i] = 1;
        }
        auto trace = m_cc->EvalMult(M, m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, batchSize));

        for (int i = 1; i <= log2(batchSize); i++) {
            m_cc->EvalAddInPlace(trace, rot.rotate(trace, batchSize / (1 << i)));
        }
        return trace;
    }

    // ============ Folding Sum for Mean Computation ============
    // Given 256×256 matrix, sum all rows
    // Result: sum replicated in all 256 rows

    Ciphertext<DCRTPoly> eval_foldingSum(Ciphertext<DCRTPoly> X, int largeDim) {
        auto result = X->Clone();

        for (int i = 1; i < largeDim; i *= 2) {
            auto rotated = rot.rotate(result, i * largeDim);
            m_cc->EvalAddInPlace(result, rotated);
        }

        return result;
    }

    // Compute mean for S_W: masked division (zeros in padding rows)
    // Result: mean in rows 0 to actualSamples-1, zeros in rows actualSamples to largeDim-1
    Ciphertext<DCRTPoly> eval_computeMeanForSw(Ciphertext<DCRTPoly> X, int actualSamples,
                                               int actualFeatures, int largeDim) {
        auto summed = eval_foldingSum(X, largeDim);

        // Create division mask:
        // rows 0 to actualSamples-1, cols 0 to actualFeatures-1: 1/actualSamples
        // everything else: 0
        std::vector<double> divMask(largeDim * largeDim, 0.0);
        for (int row = 0; row < actualSamples; row++) {
            for (int col = 0; col < actualFeatures; col++) {
                divMask[row * largeDim + col] = 1.0 / actualSamples;
            }
        }

        auto maskPtx = m_cc->MakeCKKSPackedPlaintext(divMask, 1, 0, nullptr, largeDim * largeDim);
        return m_cc->EvalMult(summed, maskPtx);
    }

    // Compute mean for S_B: scalar division, then rebatch from 256×256 to 16×16
    // Result: mean vector in 16×16 format (256 slots), mean replicated in each row
    Ciphertext<DCRTPoly> eval_computeMeanForSb(Ciphertext<DCRTPoly> X, int actualSamples,
                                               int f_tilde, int largeDim) {
        auto summed = eval_foldingSum(X, largeDim);

        // Scalar division - mean is now replicated in all 256 rows
        // Each row has [m0, m1, ..., m12, 0, ..., 0] (256 columns)
        auto meanFull = m_cc->EvalMult(summed, 1.0 / actualSamples);

        // Rebatch from 256-column rows to 16-column rows
        // Same logic as rebatchToFeatureSpace
        auto rebatched = meanFull->Clone();
        for (int i = 0; i < f_tilde - 1; i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(meanFull, (largeDim - f_tilde) * (i + 1)));
        }

        // Mask to keep only f_tilde × f_tilde elements
        std::vector<double> msk(f_tilde * f_tilde, 1.0);
        rebatched = m_cc->EvalMult(rebatched,
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, largeDim * largeDim));

        // Collapse remaining copies
        int numIterations = (int)log2((double)(largeDim * largeDim) / (f_tilde * f_tilde));
        for (int i = 0; i < numIterations; i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(rebatched, -largeDim * (1 << i)));
        }

        rebatched->SetSlots(f_tilde * f_tilde);
        return rebatched;
    }

    // ============ Outer Product for S_B ============
    // Compute (mu_c - mu)(mu_c - mu)^T
    // Input: diff vector in first f_tilde slots (from eval_computeMeanForSb)
    // Output: f_tilde × f_tilde matrix

    Ciphertext<DCRTPoly> eval_outerProduct(Ciphertext<DCRTPoly> diff, int actualF, int f_tilde) {
        // diff is in f_tilde*f_tilde slots, with meaningful values in first f_tilde positions
        // We need to compute diff * diff^T = f_tilde × f_tilde matrix

        // Step 1: Create column vector
        // Row i should have d_i replicated: [d_i, d_i, ..., d_i]
        auto colVec = getZeroCiphertext(f_tilde * f_tilde);

        for (int i = 0; i < actualF; i++) {
            // Extract d_i
            std::vector<double> extractMask(f_tilde * f_tilde, 0.0);
            extractMask[i] = 1.0;
            auto di = m_cc->EvalMult(diff,
                m_cc->MakeCKKSPackedPlaintext(extractMask, 1, 0, nullptr, f_tilde * f_tilde));

            // Replicate d_i to fill f_tilde slots
            for (int j = 1; j < f_tilde; j *= 2) {
                m_cc->EvalAddInPlace(di, rot.rotate(di, -j));
            }

            // Position at row i (mask to row i only)
            std::vector<double> rowMask(f_tilde * f_tilde, 0.0);
            for (int j = 0; j < f_tilde; j++) {
                rowMask[i * f_tilde + j] = 1.0;
            }
            auto diRow = m_cc->EvalMult(di,
                m_cc->MakeCKKSPackedPlaintext(rowMask, 1, 0, nullptr, f_tilde * f_tilde));
            m_cc->EvalAddInPlace(colVec, diRow);
        }

        // Step 2: Create row vector (diff replicated f_tilde times)
        // [d0, d1, ..., d_{f-1}, 0, ..., 0, d0, d1, ..., d_{f-1}, 0, ..., 0, ...]
        auto rowVec = diff->Clone();
        for (int i = 1; i < f_tilde; i *= 2) {
            m_cc->EvalAddInPlace(rowVec, rot.rotate(rowVec, -i * f_tilde));
        }

        // Step 3: Element-wise multiply to get outer product
        return m_cc->EvalMultAndRelinearize(colVec, rowVec);
    }

    // ============ Debug: Decrypt and Print ============

    void debugPrintLevel(const std::string& label, const Ciphertext<DCRTPoly>& cipher) {
        std::cout << "[Level] " << label << ": " << cipher->GetLevel()
                  << " / " << m_multDepth << std::endl << std::flush;
    }

    void debugPrint(const std::string& label, const Ciphertext<DCRTPoly>& cipher, int numElements = 16) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result = ptx->GetRealPackedValue();

        std::cout << "=== " << label << " (first " << numElements << " elements) ===" << std::endl;
        for (int i = 0; i < std::min(numElements, (int)result.size()); i++) {
            std::cout << std::setprecision(6) << std::fixed << result[i] << " ";
            if ((i + 1) % 8 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void debugPrintMatrix(const std::string& label, const Ciphertext<DCRTPoly>& cipher, int rows, int cols, int paddedCols) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result = ptx->GetRealPackedValue();

        std::cout << "=== " << label << " (" << rows << "x" << cols << ") [Level: "
                  << cipher->GetLevel() << "/" << m_multDepth << "] ===" << std::endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(10) << std::setprecision(4) << std::fixed
                          << result[i * paddedCols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl << std::flush;
    }

    void debugPrintVector(const std::string& label, const Ciphertext<DCRTPoly>& cipher, int len) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result = ptx->GetRealPackedValue();

        std::cout << "=== " << label << " (len=" << len << ") [Level: "
                  << cipher->GetLevel() << "/" << m_multDepth << "] ===" << std::endl;
        for (int i = 0; i < len; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << result[i] << " ";
        }
        std::cout << std::endl << std::endl << std::flush;
    }

    // ============ JKLS18 Matrix Multiplication (256×256) ============
    // This is for the large matrix multiplication X^T * X
    // Used by both AR24 and NewCol for S_W computation

    std::vector<double> generateSigmaMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d; i++) {
            msk[(i * d) + ((i + k) % d)] = 1;
        }
        return msk;
    }

    std::vector<double> generateTauMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = 0; i < d; i++) {
            msk[(((i + k) % d) * d) + i] = 1;
        }
        return msk;
    }

    // σ transform: extract k-th diagonal and shift
    Ciphertext<DCRTPoly> sigmaTransform(Ciphertext<DCRTPoly> M, int k, int d) {
        auto msk = m_cc->MakeCKKSPackedPlaintext(generateSigmaMsk(k, d), 1, 0, nullptr, d * d);
        auto result = m_cc->EvalMult(M, msk);
        // Shift so that the k-th diagonal becomes the main diagonal
        return rot.rotate(result, -k);
    }

    // τ transform: extract k-th column-diagonal and shift
    Ciphertext<DCRTPoly> tauTransform(Ciphertext<DCRTPoly> M, int k, int d) {
        auto msk = m_cc->MakeCKKSPackedPlaintext(generateTauMsk(k, d), 1, 0, nullptr, d * d);
        auto result = m_cc->EvalMult(M, msk);
        // Shift so columns align
        return rot.rotate(result, -k * d);
    }

    // JKLS18 matrix multiplication for d×d matrices
    Ciphertext<DCRTPoly> eval_mult_JKLS18(const Ciphertext<DCRTPoly>& A,
                                          const Ciphertext<DCRTPoly>& B, int d) {
        auto C = getZeroCiphertext(d * d);

        for (int k = 0; k < d; k++) {
            auto sigmaA = sigmaTransform(A, k, d);
            auto tauB = tauTransform(B, k, d);
            m_cc->EvalAddInPlace(C, m_cc->EvalMultAndRelinearize(sigmaA, tauB));
        }

        return C;
    }

    // ============ Rebatch: Extract f̃×f̃ from largeDim×largeDim matrix ============
    // After computing X^T*X in largeDim×largeDim space, extract the actual f_tilde×f_tilde result
    // Based on linear-regression rebatch logic

    Ciphertext<DCRTPoly> rebatchToFeatureSpace(Ciphertext<DCRTPoly> M,
                                               int largeDim, int f_tilde) {
        // M is largeDim×largeDim (e.g., 256×256), we want f_tilde×f_tilde (e.g., 16×16)
        // The actual data is in the top-left corner but rows are spaced largeDim apart

        // Step 1: Shift rows to be contiguous
        // Row i of result is at position i*largeDim in M, needs to be at position i*f_tilde
        auto rebatched = M->Clone();
        for (int i = 0; i < f_tilde - 1; i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(M, (largeDim - f_tilde) * (i + 1)));
        }

        // Step 2: Mask to keep only f_tilde×f_tilde elements
        std::vector<double> msk(f_tilde * f_tilde, 1.0);
        rebatched = m_cc->EvalMult(rebatched,
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, largeDim * largeDim));

        // Step 3: Collapse remaining copies using log2 additions
        int numIterations = (int)log2((double)(largeDim * largeDim) / (f_tilde * f_tilde));
        for (int i = 0; i < numIterations; i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(rebatched, -largeDim * (1 << i)));
        }

        rebatched->SetSlots(f_tilde * f_tilde);
        return rebatched;
    }

public:
    LDAEncryptedBase(std::shared_ptr<Encryption> enc,
                     CryptoContext<DCRTPoly> cc,
                     KeyPair<DCRTPoly> keyPair,
                     std::vector<int> rotIndices,
                     int multDepth,
                     bool useBootstrapping = true)
        : m_enc(enc)
        , m_cc(cc)
        , m_keyPair(keyPair)
        , m_rotIndices(rotIndices)
        , rot(cc, rotIndices, cc->GetRingDimension() / 2)
        , m_multDepth(multDepth)
        , m_useBootstrapping(useBootstrapping)
    {}

    virtual ~LDAEncryptedBase() = default;

    // Pure virtual: each algorithm implements its own inversion
    virtual Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d, int iterations) = 0;

    // Main training function (common workflow)
    virtual LDAEncryptedResult trainWithTimings(
        const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
        const LDADataset& dataset,
        int inversionIterations,
        LDATimingResult& timings,
        bool verbose = false) = 0;

    void setBootstrapping(bool enable) { m_useBootstrapping = enable; }
    bool getBootstrapping() const { return m_useBootstrapping; }
};
