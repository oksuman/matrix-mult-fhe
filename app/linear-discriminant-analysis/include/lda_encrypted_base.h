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
    std::vector<double> Sw_inv_decrypted;
    std::vector<size_t> classCounts;
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
    // Given vector packed as s̃ rows of f̃ features, sum all rows
    // Result: sum replicated s̃ times

    Ciphertext<DCRTPoly> eval_foldingSum(Ciphertext<DCRTPoly> X, int paddedSamples, int paddedFeatures) {
        auto result = X->Clone();
        int rowSize = paddedFeatures;

        // log2(paddedSamples) rotations and additions
        for (int i = 1; i < paddedSamples; i *= 2) {
            auto rotated = rot.rotate(result, i * rowSize);
            m_cc->EvalAddInPlace(result, rotated);
        }

        return result;
    }

    // Compute mean: folding sum + division by actual sample count
    Ciphertext<DCRTPoly> eval_computeMean(Ciphertext<DCRTPoly> X, int actualSamples,
                                          int paddedSamples, int paddedFeatures) {
        auto summed = eval_foldingSum(X, paddedSamples, paddedFeatures);
        return m_cc->EvalMult(summed, 1.0 / actualSamples);
    }

    // ============ Replicate Mean for Broadcasting ============
    // Given mean in first f̃ slots, replicate to fill targetSlots
    // Uses SetSlots for sparse packing replication

    Ciphertext<DCRTPoly> eval_replicateMean(Ciphertext<DCRTPoly> mean, int paddedFeatures, int targetSlots) {
        // First extract just the first row (f̃ elements)
        std::vector<double> rowMask(paddedFeatures, 1.0);
        auto singleRow = m_cc->EvalMult(mean,
            m_cc->MakeCKKSPackedPlaintext(rowMask, 1, 0, nullptr, paddedFeatures));

        // Use SetSlots to expand to target size (automatic replication in CKKS sparse packing)
        singleRow->SetSlots(targetSlots);

        // Clean up: ensure proper replication pattern
        // After SetSlots, the f̃ values get replicated, but we need row-wise replication
        // So we need to use rotation and masking to achieve the right pattern

        // Alternative approach: sum shifted copies
        auto result = getZeroCiphertext(targetSlots);
        int numReplicates = targetSlots / paddedFeatures;

        for (int i = 0; i < numReplicates; i++) {
            // Shift mean to position i*paddedFeatures
            std::vector<double> posMask(targetSlots, 0.0);
            for (int j = 0; j < paddedFeatures; j++) {
                posMask[i * paddedFeatures + j] = 1.0;
            }
            auto shifted = m_cc->EvalMult(rot.rotate(singleRow, -i * paddedFeatures),
                m_cc->MakeCKKSPackedPlaintext(posMask, 1, 0, nullptr, targetSlots));
            m_cc->EvalAddInPlace(result, shifted);
        }

        return result;
    }

    // ============ Outer Product for S_B ============
    // Compute (mu_c - mu)(mu_c - mu)^T using replicated vectors
    // Input: diff vector of length f (in first f̃ slots), replicated
    // Output: f×f matrix (padded to f̃×f̃)

    Ciphertext<DCRTPoly> eval_outerProduct(Ciphertext<DCRTPoly> diff, int actualF, int paddedF) {
        // diff is assumed to be in first paddedF slots: [d0, d1, ..., d_{f-1}, 0, ..., 0]
        // We need to compute diff * diff^T

        // Step 1: Create replicated column vector: each d_i repeated f̃ times consecutively
        // [d0, d0, d0, ..., d1, d1, d1, ..., d_{f-1}, d_{f-1}, ...]
        auto colVec = getZeroCiphertext(paddedF * paddedF);
        for (int i = 0; i < actualF; i++) {
            // Extract d_i and replicate paddedF times starting at position i*paddedF
            std::vector<double> extractMask(paddedF, 0.0);
            extractMask[i] = 1.0;
            auto di = m_cc->EvalMult(diff,
                m_cc->MakeCKKSPackedPlaintext(extractMask, 1, 0, nullptr, paddedF));

            // Sum to replicate within paddedF slots
            for (int j = 1; j < paddedF; j *= 2) {
                m_cc->EvalAddInPlace(di, rot.rotate(di, -j));
            }

            // Position at row i
            std::vector<double> rowMask(paddedF * paddedF, 0.0);
            for (int j = 0; j < paddedF; j++) {
                rowMask[i * paddedF + j] = 1.0;
            }
            di->SetSlots(paddedF * paddedF);
            auto diRow = m_cc->EvalMult(di,
                m_cc->MakeCKKSPackedPlaintext(rowMask, 1, 0, nullptr, paddedF * paddedF));
            m_cc->EvalAddInPlace(colVec, diRow);
        }

        // Step 2: Create replicated row vector: diff repeated f̃ times
        // [d0, d1, ..., d_{f-1}, 0, ..., d0, d1, ..., d_{f-1}, 0, ..., ...]
        diff->SetSlots(paddedF * paddedF);
        auto rowVec = diff->Clone();
        for (int i = 1; i < paddedF; i *= 2) {
            m_cc->EvalAddInPlace(rowVec, rot.rotate(rowVec, -i * paddedF));
        }

        // Step 3: Element-wise multiply to get outer product
        return m_cc->EvalMultAndRelinearize(colVec, rowVec);
    }

    // ============ Debug: Decrypt and Print ============

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

    // ============ Rebatch: Extract f×f from larger matrix ============
    // After computing X^T*X in d×d space, extract the actual f̃×f̃ result

    Ciphertext<DCRTPoly> rebatchToFeatureSpace(Ciphertext<DCRTPoly> M,
                                               int largeDim, int smallDim) {
        // M is largeDim×largeDim, we want to extract smallDim×smallDim
        // The actual data is in the top-left corner

        // First, sum along the rows to collect the smallDim columns
        auto rebatched = M->Clone();
        for (int i = 0; i < smallDim - 1; i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(M, (largeDim - smallDim) * (i + 1)));
        }

        // Mask to keep only smallDim×smallDim
        std::vector<double> msk(smallDim * smallDim, 1.0);
        rebatched = m_cc->EvalMult(rebatched,
            m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, largeDim * largeDim));

        // Compact the rows
        for (int i = 0; i < (int)log2((largeDim * largeDim) / (smallDim * smallDim)); i++) {
            m_cc->EvalAddInPlace(rebatched,
                rot.rotate(rebatched, -largeDim * (1 << i)));
        }

        rebatched->SetSlots(smallDim * smallDim);
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
