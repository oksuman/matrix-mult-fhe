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

    // Compute mean for S_B: row-wise folding sum, then column-wise folding sum
    // Result: 16-replicated form in f_tilde×f_tilde slots
    // Each row of the 16×16 matrix has [m0, m1, ..., m_{f-1}, 0, ..., 0]
    Ciphertext<DCRTPoly> eval_computeMeanForSb(Ciphertext<DCRTPoly> X, int actualSamples,
                                               int f_tilde, int largeDim) {
        // Step 1: Row-wise folding sum (sum replicated in all 256 rows)
        auto summed = eval_foldingSum(X, largeDim);

        // Step 2: Divide by number of samples
        // Now each of 256 rows has [m0, m1, ..., m_{f-1}, 0, ..., 0] (256 columns)
        auto mean = m_cc->EvalMult(summed, 1.0 / actualSamples);

        // Step 3: Column-wise folding sum to replicate mean vector within each row
        // Rotations: -f_tilde, -2*f_tilde, -4*f_tilde, -8*f_tilde (i.e., -16, -32, -64, -128)
        // This creates [m0..m15] repeated 16 times per row (256 = 16 * 16)
        for (int i = f_tilde; i < largeDim; i *= 2) {
            m_cc->EvalAddInPlace(mean, rot.rotate(mean, -i));
        }

        // Step 4: SetSlots to interpret as 16×16 matrix
        // First 256 slots form a 16×16 matrix where each row is [m0, m1, ..., m15]
        // This is the 16-replicated form
        mean->SetSlots(f_tilde * f_tilde);

        return mean;
    }

    // Compute global mean from encrypted class means (weighted average)
    // μ = (n_0 * μ_0 + n_1 * μ_1 + ...) / (n_0 + n_1 + ...)
    // All inputs and output are in 16-replicated form
    Ciphertext<DCRTPoly> eval_computeGlobalMean(
        const std::vector<Ciphertext<DCRTPoly>>& classMeans,
        const std::vector<size_t>& classCounts,
        int f_tilde) {

        size_t totalSamples = 0;
        for (auto n : classCounts) totalSamples += n;

        // Weighted sum: sum_c (n_c * μ_c)
        auto globalMean = m_cc->EvalMult(classMeans[0], (double)classCounts[0]);
        for (size_t c = 1; c < classMeans.size(); c++) {
            auto scaled = m_cc->EvalMult(classMeans[c], (double)classCounts[c]);
            m_cc->EvalAddInPlace(globalMean, scaled);
        }

        // Divide by total samples
        globalMean = m_cc->EvalMult(globalMean, 1.0 / totalSamples);
        globalMean->SetSlots(f_tilde * f_tilde);

        return globalMean;
    }

    // ============ Outer Product for S_B ============
    // Compute (mu_c - mu)(mu_c - mu)^T using transpose + Hadamard product
    // Input: diff in 16-replicated form (each row is [d0, d1, ..., d_{f-1}, 0, ..., 0])
    // Output: f_tilde × f_tilde outer product matrix
    //
    // diff:   position (i, j) = d_j  (row-replicated: each row is the diff vector)
    // diff^T: position (i, j) = d_i  (column-replicated: each column is the diff vector)
    // diff * diff^T (Hadamard): position (i, j) = d_i * d_j  (outer product!)

    Ciphertext<DCRTPoly> eval_outerProduct(Ciphertext<DCRTPoly> diff, int actualF, int f_tilde) {
        // diff is in 16-replicated form (f_tilde × f_tilde slots)
        // Transpose to get column-replicated form
        auto diff_T = eval_transpose(diff, f_tilde, f_tilde * f_tilde);

        // Hadamard product gives outer product
        return m_cc->EvalMultAndRelinearize(diff, diff_T);
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
        // ============================================================
        // Rebatch: Extract f̃×f̃ matrix from largeDim×largeDim space
        // ============================================================
        // Input: M is largeDim×largeDim (e.g., 256×256 = 65536 slots)
        //        Actual data is in top-left f_tilde×f_tilde (e.g., 16×16)
        //        Row i is at position i*largeDim (stride = largeDim)
        //
        // Output: f̃×f̃ matrix replicated s times for AR24 algorithm
        //         Total slots = f̃² × s (e.g., 256 × 16 = 4096)
        //
        // Steps:
        //   1. Row folding: change stride from largeDim to f_tilde
        //   2. Masking: keep only first f̃² slots
        //   3. Replication: copy matrix s times for AR24
        // ============================================================

        int gap = largeDim - f_tilde;  // 256 - 16 = 240

        // Compute replication count s for AR24 algorithm
        int ringDim = m_cc->GetRingDimension();
        int s = std::min(f_tilde, ringDim / 2 / f_tilde / f_tilde);
        s = std::max(1, s);  // At least 1
        int num_slots = f_tilde * f_tilde * s;  // 16 * 16 * 16 = 4096

        // Step 1: Log-depth row folding (log2(f_tilde) = 4 rotations)
        // Rotation amounts: gap, 2*gap, 4*gap, 8*gap = 240, 480, 960, 1920
        // After folding: rows 0-15 are contiguous in positions 0-255
        auto rebatched = M->Clone();
        for (int step = 1; step < f_tilde; step *= 2) {
            auto rotated = rot.rotate(rebatched, step * gap);
            m_cc->EvalAddInPlace(rebatched, rotated);
        }

        // Step 2: Mask to keep only f̃×f̃ elements (positions 0 to f̃²-1)
        // This removes garbage/duplicates in positions f̃² and beyond
        std::vector<double> msk(f_tilde * f_tilde, 1.0);
        rebatched = m_cc->EvalMult(rebatched,
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, largeDim * largeDim));

        // Step 3: Replicate matrix s times for AR24 algorithm
        // Rotation amounts: -f̃², -2f̃², -4f̃², -8f̃² = -256, -512, -1024, -2048
        // After replication: s copies of f̃×f̃ matrix in positions 0 to s×f̃²-1
        for (int i = 1; i < s; i *= 2) {
            auto rotated = rot.rotate(rebatched, -i * f_tilde * f_tilde);
            m_cc->EvalAddInPlace(rebatched, rotated);
        }

        rebatched->SetSlots(num_slots);
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
    // Client sends: classDataEncrypted (per-class) + sample counts (plaintext)
    // Global mean is computed from class means (weighted average)
    // sbOnly: if true, stop after S_B computation (for quick testing)
    virtual LDAEncryptedResult trainWithTimings(
        const std::vector<Ciphertext<DCRTPoly>>& classDataEncrypted,
        const LDADataset& dataset,
        int inversionIterations,
        LDATimingResult& timings,
        bool verbose = false,
        bool sbOnly = false) = 0;

    void setBootstrapping(bool enable) { m_useBootstrapping = enable; }
    bool getBootstrapping() const { return m_useBootstrapping; }
};
