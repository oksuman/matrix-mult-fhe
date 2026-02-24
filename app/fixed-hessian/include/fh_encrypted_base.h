// fh_encrypted_base.h
// Base class for encrypted logistic regression (Fixed Hessian)
// Contains: JKLS18 multiplication, transpose, trace, scalar inverse,
//           rebatch, computeXty, matVecMult, debug utilities
#pragma once

#include "encryption.h"
#include "rotation.h"
#include "fh_data_encoder.h"
#include <map>
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <set>
#include <cmath>

class FHEncryptedBase {
protected:
    std::shared_ptr<Encryption> m_enc;
    CryptoContext<DCRTPoly> m_cc;
    KeyPair<DCRTPoly> m_keyPair;
    std::vector<int> m_rotIndices;
    RotationComposer rot;
    int m_multDepth;
    bool m_useBootstrapping;
    bool m_verbose;

    // ============ Utility Functions (protected) ============

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

    Ciphertext<DCRTPoly> getZeroCiphertext(int batchSize) {
        std::vector<double> zeroVec(batchSize, 0.0);
        auto zeroPtx = m_cc->MakeCKKSPackedPlaintext(zeroVec, 1, 0, nullptr, batchSize);
        return m_cc->Encrypt(zeroPtx, m_keyPair.publicKey);
    }

    mutable std::map<int, Ciphertext<DCRTPoly>> m_zeroCache;

    Ciphertext<DCRTPoly> makeZero(int batchSize) {
        auto it = m_zeroCache.find(batchSize);
        if (it == m_zeroCache.end()) {
            m_zeroCache[batchSize] = getZeroCiphertext(batchSize);
        }
        return m_zeroCache.at(batchSize)->Clone();
    }

    // ============ JKLS18 helpers (protected) ============

    std::vector<double> generateSigmaMsk(int k, int d) {
        std::vector<double> u(d * d, 0);
        if (k >= 0) {
            for (int i = d * k; i < d - k + d * k && i < d * d; ++i) {
                u[i] = 1.0;
            }
        } else {
            for (int i = 0; i < d * d; i++) {
                if (i < d + d * (d + k) && i >= -k + d * (d + k))
                    u[i] = 1.0;
            }
        }
        return u;
    }

    std::vector<double> generateTauMsk(int k, int d) {
        std::vector<double> msk(d * d, 0);
        for (int i = k; i < d * d; i += d)
            msk[i] = 1;
        return msk;
    }

    std::vector<double> generateShiftingMsk(int k, int d) {
        std::vector<double> v(d * d, 0);
        for (int i = k; i < d * d; i += d) {
            for (int j = i; j < i + d - k; ++j) {
                v[j] = 1;
            }
        }
        return v;
    }

    Ciphertext<DCRTPoly> columnShifting(const Ciphertext<DCRTPoly>& M, int l, int d) {
        if (l == 0) return M->Clone();
        std::vector<double> msk = generateShiftingMsk(l, d);
        Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
        auto tmp = m_cc->EvalMult(pmsk, M);
        auto M_1 = rot.rotate(m_cc->EvalSub(M, tmp), l - d);
        auto M_2 = rot.rotate(tmp, l);
        return m_cc->EvalAdd(M_1, M_2);
    }

    Ciphertext<DCRTPoly> sigmaTransform(const Ciphertext<DCRTPoly>& M, int d) {
        auto sigma_M = makeZero(d * d);

        // For d=128, use bs=8, gs=16 instead of bs=11
        int bs = (d == 128) ? 8 : (int)round(sqrt((double)d));
        int gs = d / bs;

        std::vector<Ciphertext<DCRTPoly>> babySteps(bs);
        for (int i = 0; i < bs; i++) {
            babySteps[i] = rot.rotate(M, i);
        }

        for (int i = 1; i < d - bs * (gs - 1); i++) {
            Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                generateSigmaMsk(-d + i, d), 1, 0, nullptr, d * d);
            m_cc->EvalAddInPlace(sigma_M,
                m_cc->EvalMult(rot.rotate(M, i - d), pmsk));
        }

        for (int i = -(gs - 1); i < gs; i++) {
            auto tmp = makeZero(d * d);
            for (int j = 0; j < bs; j++) {
                auto msk = generateSigmaMsk(bs * i + j, d);
                msk = vectorRotate(msk, -bs * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(pmsk, babySteps[j]));
            }
            m_cc->EvalAddInPlace(sigma_M, rot.rotate(tmp, bs * i));
        }

        return sigma_M;
    }

    Ciphertext<DCRTPoly> tauTransform(const Ciphertext<DCRTPoly>& M, int d) {
        auto tau_M = makeZero(d * d);

        double squareRootd = sqrt((double)d);
        int squareRootIntd = (int)squareRootd;

        if (squareRootIntd * squareRootIntd == d) {
            // Perfect square case (d=64: bs=gs=8)
            std::vector<Ciphertext<DCRTPoly>> babySteps(squareRootIntd);
            for (int i = 0; i < squareRootIntd; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < squareRootIntd; i++) {
                auto tmp = makeZero(d * d);
                for (int j = 0; j < squareRootIntd; j++) {
                    auto msk = generateTauMsk(squareRootIntd * i + j, d);
                    msk = vectorRotate(msk, -squareRootIntd * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                    m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, squareRootIntd * d * i));
            }
        } else if (d == 128) {
            // Special case for d=128: bs=8, gs=16
            int bs = 8;
            int gs = 16;

            std::vector<Ciphertext<DCRTPoly>> babySteps(bs);
            for (int i = 0; i < bs; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            // Handle extra diagonals: bs*(gs-1) to d-1 = 120 to 127
            for (int i = 0; i < d - bs * (gs - 1); i++) {
                Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                    generateTauMsk(bs * (gs - 1) + i, d), 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tau_M,
                    m_cc->EvalMult(rot.rotate(M, (bs * (gs - 1) + i) * d), pmsk));
            }

            // Main loop: diagonals 0 to bs*(gs-1)-1 = 0 to 119
            for (int i = 0; i < gs - 1; i++) {
                auto tmp = makeZero(d * d);
                for (int j = 0; j < bs; j++) {
                    auto msk = generateTauMsk(bs * i + j, d);
                    msk = vectorRotate(msk, -bs * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                    m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, bs * d * i));
            }
        } else {
            // General non-perfect-square case
            int steps = (int)round(squareRootd);

            std::vector<Ciphertext<DCRTPoly>> babySteps(steps);
            for (int i = 0; i < steps; i++) {
                babySteps[i] = rot.rotate(M, d * i);
            }

            for (int i = 0; i < d - steps * (steps - 1); i++) {
                Plaintext pmsk = m_cc->MakeCKKSPackedPlaintext(
                    generateTauMsk(steps * (steps - 1) + i, d), 1, 0, nullptr, d * d);
                m_cc->EvalAddInPlace(tau_M,
                    m_cc->EvalMult(rot.rotate(M, (steps * (steps - 1) + i) * d), pmsk));
            }

            for (int i = 0; i < steps - 1; i++) {
                auto tmp = makeZero(d * d);
                for (int j = 0; j < steps; j++) {
                    auto msk = generateTauMsk(steps * i + j, d);
                    msk = vectorRotate(msk, -steps * d * i);
                    auto pmsk = m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, d * d);
                    m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babySteps[j], pmsk));
                }
                m_cc->EvalAddInPlace(tau_M, rot.rotate(tmp, steps * d * i));
            }
        }

        return tau_M;
    }

    // ============ Transpose mask (protected) ============

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

public:
    FHEncryptedBase(std::shared_ptr<Encryption> enc,
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
        , m_verbose(false)
    {}

    virtual ~FHEncryptedBase() = default;

    void setVerbose(bool v) { m_verbose = v; }
    void setBootstrapping(bool enable) { m_useBootstrapping = enable; }

    // Pure virtual: each algorithm implements its own inversion
    virtual Ciphertext<DCRTPoly> eval_inverse(const Ciphertext<DCRTPoly>& M, int d,
                                               int iterations, int actualDim,
                                               double traceUpperBound) = 0;

    // ============ Public methods (called from main) ============

    Ciphertext<DCRTPoly> eval_transpose(Ciphertext<DCRTPoly> M, int d, int batchSize) {
        // Baby-step giant-step: bs * gs = d
        // d=64: bs=8, gs=8
        // d=128: bs=8, gs=16
        int bs = 8;
        int gs = d / bs;

        std::vector<Ciphertext<DCRTPoly>> babyStepsOfM(bs);
        for (int i = 0; i < bs; i++) {
            babyStepsOfM[i] = rot.rotate(M, (d - 1) * i);
        }

        auto M_transposed = makeZero(batchSize);

        for (int i = -gs; i < gs; i++) {
            auto tmp = makeZero(batchSize);

            int js = (i == -gs) ? 1 : 0;
            for (int j = js; j < bs; j++) {
                int k = bs * i + j;
                if (k >= d || k <= -d) continue;

                auto vmsk = generateTransposeMask(k, d);
                vmsk = vectorRotate(vmsk, -bs * (d - 1) * i);
                auto pmsk = m_cc->MakeCKKSPackedPlaintext(vmsk, 1, 0, nullptr, batchSize);
                m_cc->EvalAddInPlace(tmp, m_cc->EvalMult(babyStepsOfM[j], pmsk));
            }

            tmp = rot.rotate(tmp, bs * (d - 1) * i);
            m_cc->EvalAddInPlace(M_transposed, tmp);
        }

        return M_transposed;
    }

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

    Ciphertext<DCRTPoly> eval_scalar_inverse(const Ciphertext<DCRTPoly>& t, double upperBound,
                                              int iterations, int batchSize) {
        double x0 = 1.0 / upperBound;
        auto x = m_cc->Encrypt(m_keyPair.publicKey,
            m_cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, x0), 1, 0, nullptr, batchSize));
        auto t_bar = m_cc->EvalSub(1.0, m_cc->EvalMult(t, x0));

        if (m_verbose) {
            std::cout << "  [Scalar Inv] upper_bound = " << upperBound << ", x0 = " << x0 << std::endl;
        }

        for (int i = 0; i < iterations; i++) {
            x = m_cc->EvalMult(x, m_cc->EvalAdd(t_bar, 1.0));
            t_bar = m_cc->EvalMult(t_bar, t_bar);
        }
        return x;
    }

    Ciphertext<DCRTPoly> eval_foldingSum(Ciphertext<DCRTPoly> X, int d) {
        auto result = X->Clone();
        for (int i = 1; i < d; i *= 2) {
            auto rotated = rot.rotate(result, i * d);
            m_cc->EvalAddInPlace(result, rotated);
        }
        return result;
    }

    Ciphertext<DCRTPoly> eval_mult_JKLS18(const Ciphertext<DCRTPoly>& A,
                                          const Ciphertext<DCRTPoly>& B, int d) {
        auto sigma_A = sigmaTransform(A, d);
        auto tau_B = tauTransform(B, d);
        auto matrixC = m_cc->EvalMultAndRelinearize(sigma_A, tau_B);

        for (int i = 1; i < d; i++) {
            auto shifted_A = columnShifting(sigma_A, i, d);
            tau_B = rot.rotate(tau_B, d);
            m_cc->EvalAddInPlace(matrixC,
                m_cc->EvalMultAndRelinearize(shifted_A, tau_B));
        }

        return matrixC;
    }

    Ciphertext<DCRTPoly> rebatchToFeatureSpace(Ciphertext<DCRTPoly> M,
                                               int largeDim, int f_tilde) {
        int gap = largeDim - f_tilde;
        int ringDim = m_cc->GetRingDimension();
        int s = std::min(f_tilde, ringDim / 2 / f_tilde / f_tilde);
        s = std::max(1, s);
        int num_slots = f_tilde * f_tilde * s;

        int rowsPerChunk = largeDim / f_tilde;

        Ciphertext<DCRTPoly> rebatched;

        if (rowsPerChunk >= f_tilde) {
            rebatched = M->Clone();
            for (int step = 1; step < f_tilde; step *= 2) {
                auto rotated = rot.rotate(rebatched, step * gap);
                m_cc->EvalAddInPlace(rebatched, rotated);
            }

            std::vector<double> msk(f_tilde * f_tilde, 1.0);
            rebatched = m_cc->EvalMult(rebatched,
                m_cc->MakeCKKSPackedPlaintext(msk, 1, 0, nullptr, largeDim * largeDim));
        } else {
            int numChunks = (f_tilde + rowsPerChunk - 1) / rowsPerChunk;
            std::vector<Ciphertext<DCRTPoly>> chunkResults;

            for (int chunk = 0; chunk < numChunks; chunk++) {
                int startRow = chunk * rowsPerChunk;
                int endRow = std::min(startRow + rowsPerChunk, f_tilde);

                std::vector<double> chunkMask(largeDim * largeDim, 0.0);
                for (int row = startRow; row < endRow; row++) {
                    for (int col = 0; col < f_tilde; col++) {
                        chunkMask[row * f_tilde + col] = 1.0;
                    }
                }

                auto chunkData = M->Clone();
                for (int step = 1; step < rowsPerChunk && step < (endRow - startRow); step *= 2) {
                    auto rotated = rot.rotate(chunkData, step * gap);
                    m_cc->EvalAddInPlace(chunkData, rotated);
                }

                if (startRow > 0) {
                    int rotAmount = startRow * largeDim - startRow * f_tilde;
                    chunkData = rot.rotate(chunkData, rotAmount);
                }

                chunkData = m_cc->EvalMult(chunkData,
                    m_cc->MakeCKKSPackedPlaintext(chunkMask, 1, 0, nullptr, largeDim * largeDim));

                chunkResults.push_back(chunkData);
            }

            rebatched = chunkResults[0];
            for (size_t i = 1; i < chunkResults.size(); i++) {
                m_cc->EvalAddInPlace(rebatched, chunkResults[i]);
            }

            // For rowsPerChunk < f_tilde case, need more replication
            // to fill largeDim*largeDim/2 slots before SetSlots
            int s_large = largeDim * largeDim / (f_tilde * f_tilde);
            for (int i = 1; i < s_large; i *= 2) {
                auto rotated = rot.rotate(rebatched, -i * f_tilde * f_tilde);
                m_cc->EvalAddInPlace(rebatched, rotated);
            }

            rebatched->SetSlots(num_slots);
            return rebatched;
        }

        // Replicate matrix s times (for rowsPerChunk >= f_tilde case)
        for (int i = 1; i < s; i *= 2) {
            auto rotated = rot.rotate(rebatched, -i * f_tilde * f_tilde);
            m_cc->EvalAddInPlace(rebatched, rotated);
        }

        rebatched->SetSlots(num_slots);
        return rebatched;
    }

    Ciphertext<DCRTPoly> computeXty(const Ciphertext<DCRTPoly>& X,
                                    const Ciphertext<DCRTPoly>& y,
                                    int featureDim, int sampleDim) {
        auto y_replicated = y->Clone();
        y_replicated->SetSlots(sampleDim * sampleDim);

        auto y_transposed = eval_transpose(y_replicated, sampleDim, sampleDim * sampleDim);

        auto result = m_cc->EvalMultAndRelinearize(X, y_transposed);

        // Row folding sum
        for (int i = 0; i < (int)log2(sampleDim); i++) {
            m_cc->EvalAddInPlace(result, rot.rotate(result, sampleDim * (1 << i)));
        }

        result->SetSlots(sampleDim);

        // Fold down to featureDim
        for (int i = 0; i < (int)log2(sampleDim / featureDim); i++) {
            m_cc->EvalAddInPlace(result, rot.rotate(result, featureDim * (1 << i)));
        }

        return result;
    }

    Ciphertext<DCRTPoly> matVecMult_enc(const Ciphertext<DCRTPoly>& M_enc,
                                        const Ciphertext<DCRTPoly>& v_enc,
                                        int d) {
        auto v_rep = v_enc->Clone();
        v_rep->SetSlots(d * d);
        auto v_T = eval_transpose(v_rep, d, d * d);
        auto product = m_cc->EvalMultAndRelinearize(M_enc, v_T);

        for (int i = 0; i < (int)log2(d); i++) {
            int shift = d * (1 << i);
            m_cc->EvalAddInPlace(product, rot.rotate(product, shift));
        }
        return product;
    }

    Ciphertext<DCRTPoly> matVecMult_plain(const Ciphertext<DCRTPoly>& M_enc,
                                          const std::vector<double>& v_plain,
                                          int d) {
        // Build column-replicated plaintext: position(i,j) = v[i]
        std::vector<double> v_T(d * d, 0.0);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                v_T[i * d + j] = v_plain[i];
            }
        }

        auto v_ptx = m_cc->MakeCKKSPackedPlaintext(v_T, 1, 0, nullptr, d * d);
        auto product = m_cc->EvalMult(M_enc, v_ptx);

        // Row folding sum
        for (int i = 0; i < (int)log2(d); i++) {
            int shift = d * (1 << i);
            m_cc->EvalAddInPlace(product, rot.rotate(product, shift));
        }
        return product;
    }

    // ============ Simplified Fixed Hessian (public) ============

    // Compute diagonal Hessian elements from encrypted X (64x64)
    // H̃(j)(j) = -1/4 * Σᵢ(x_ij * Σₖ x_ik)
    // Returns vector of 16 diagonal elements (in sampleDim-length ciphertext)
    Ciphertext<DCRTPoly> computeDiagHessian_enc(const Ciphertext<DCRTPoly>& X,
                                                 int sampleDim, int featureDim) {
        int batchSize = sampleDim * sampleDim;

        // 1. Column folding sum: position(i,0) = Σₖ x_ik (sum of row i)
        auto colFolded = X->Clone();
        for (int i = 1; i < sampleDim; i *= 2) {
            m_cc->EvalAddInPlace(colFolded, rot.rotate(colFolded, i));
        }

        // 2. Mask column 0 only
        std::vector<double> col0Mask(batchSize, 0.0);
        for (int i = 0; i < sampleDim; i++) {
            col0Mask[i * sampleDim] = 1.0;
        }
        auto masked = m_cc->EvalMult(colFolded,
            m_cc->MakeCKKSPackedPlaintext(col0Mask, 1, 0, nullptr, batchSize));

        // 3. Replicate column 0 to all columns
        auto replicated = masked->Clone();
        for (int i = 1; i < sampleDim; i *= 2) {
            m_cc->EvalAddInPlace(replicated, rot.rotate(replicated, -i));
        }

        // 4. Hadamard(X, replicated): position(i,j) = x_ij * Σₖ x_ik
        auto product = m_cc->EvalMultAndRelinearize(X, replicated);

        // 5. Row folding sum: position(0,j) = Σᵢ(x_ij * Σₖ x_ik)
        for (int i = 1; i < sampleDim; i *= 2) {
            m_cc->EvalAddInPlace(product, rot.rotate(product, i * sampleDim));
        }

        // 6. Multiply by -1/4 to get H̃(j)(j)
        auto diagH = m_cc->EvalMult(product, -0.25);

        // Result is row-replicated (all rows same) in 128*128 slots.
        // First SetSlots(256) to get 16x16 structure (rows 0 and 1 interleaved).
        // At this point: slot 0~127 = row 0, slot 128~255 = row 1 (same values)
        diagH->SetSlots(featureDim * featureDim);  // 256

        // Mask to keep only slot 0~15 (first row, first 16 elements)
        std::vector<double> featureMask(featureDim * featureDim, 0.0);
        for (int j = 0; j < featureDim; j++) {
            featureMask[j] = 1.0;
        }
        diagH = m_cc->EvalMult(diagH,
            m_cc->MakeCKKSPackedPlaintext(featureMask, 1, 0, nullptr, featureDim * featureDim));

        // Fold to make 16-replicated: [d0..d15, d0..d15, ...]
        for (int i = featureDim; i < featureDim * featureDim; i *= 2) {
            m_cc->EvalAddInPlace(diagH, rot.rotate(diagH, -i));
        }
        return diagH;
    }

    // Iterative diagonal inverse: x_{n+1} = 2*x_n - a*x_n^2
    Ciphertext<DCRTPoly> eval_diagonal_inverse(
                            const Ciphertext<DCRTPoly>& diagH,
                            int numSamples, int featureDim, int sampleDim,
                            int iterations = 2) {
        int actualFeatures = FH_RAW_FEATURES + 1;
        double u0 = -16.0 / (numSamples * actualFeatures);

        if (m_verbose) {
            std::cout << "  [DiagInverse] u0 = " << u0
                      << " (N=" << numSamples << ", f=" << featureDim
                      << ", iterations=" << iterations << ")" << std::endl;
        }

        // First iteration: x_1 = 2*u0 - diagH * u0^2
        double twoU0 = 2.0 * u0;
        double u0Sq = u0 * u0;

        auto diagH_u0sq = m_cc->EvalMult(diagH, u0Sq);
        int slots = featureDim * featureDim;
        std::vector<double> twoU0Vec(slots, twoU0);
        auto twoU0Ptx = m_cc->MakeCKKSPackedPlaintext(twoU0Vec, 1, 0, nullptr, slots);
        auto invDiag = m_cc->EvalSub(twoU0Ptx, diagH_u0sq);

        if (m_verbose) {
            Plaintext ptxInv;
            m_cc->Decrypt(m_keyPair.secretKey, invDiag, &ptxInv);
            auto invVec = ptxInv->GetRealPackedValue();
            std::cout << "  [DiagInverse iter 1] inv_diag (first 8): ";
            for (int i = 0; i < std::min(8, featureDim); i++) {
                std::cout << std::setprecision(6) << invVec[i] << " ";
            }
            std::cout << std::endl;
        }

        // Additional iterations: x_{n+1} = 2*x_n - diagH * x_n^2
        for (int iter = 1; iter < iterations; iter++) {
            auto invDiagSq = m_cc->EvalMultAndRelinearize(invDiag, invDiag);
            auto diagH_xSq = m_cc->EvalMultAndRelinearize(diagH, invDiagSq);
            auto twoX = m_cc->EvalMult(invDiag, 2.0);
            invDiag = m_cc->EvalSub(twoX, diagH_xSq);

            if (m_verbose) {
                Plaintext ptxInv;
                m_cc->Decrypt(m_keyPair.secretKey, invDiag, &ptxInv);
                auto invVec = ptxInv->GetRealPackedValue();
                std::cout << "  [DiagInverse iter " << (iter + 1) << "] inv_diag (first 8): ";
                for (int i = 0; i < std::min(8, featureDim); i++) {
                    std::cout << std::setprecision(6) << invVec[i] << " ";
                }
                std::cout << std::endl;
            }
        }

        return invDiag;
    }

    // ============ Debug Utilities (public) ============

    void debugPrintLevel(const std::string& label, const Ciphertext<DCRTPoly>& cipher) {
        std::cout << "[Level] " << label << ": " << cipher->GetLevel()
                  << " / " << m_multDepth << std::endl << std::flush;
    }

    void debugPrintMatrix(const std::string& label, const Ciphertext<DCRTPoly>& cipher,
                         int rows, int cols, int paddedCols) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result = ptx->GetRealPackedValue();

        int showRows = std::min(rows, 8);
        int showCols = std::min(cols, 8);

        std::cout << "=== " << label << " (" << rows << "x" << cols << ") [Level: "
                  << cipher->GetLevel() << "/" << m_multDepth << "] ===" << std::endl;
        for (int i = 0; i < showRows; i++) {
            for (int j = 0; j < showCols; j++) {
                std::cout << std::setw(12) << std::setprecision(6) << std::fixed
                          << result[i * paddedCols + j];
            }
            if (cols > 8) std::cout << " ...";
            std::cout << std::endl;
        }
        if (rows > 8) std::cout << "  ..." << std::endl;
        std::cout << std::endl << std::flush;
    }

    void debugPrintVector(const std::string& label, const Ciphertext<DCRTPoly>& cipher, int len) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        std::vector<double> result = ptx->GetRealPackedValue();

        int show = std::min(len, 16);
        std::cout << "=== " << label << " (len=" << len << ") [Level: "
                  << cipher->GetLevel() << "/" << m_multDepth << "] ===" << std::endl;
        for (int i = 0; i < show; i++) {
            std::cout << std::setw(10) << std::setprecision(4) << std::fixed << result[i] << " ";
        }
        std::cout << std::endl << std::endl << std::flush;
    }

    double debugGetTrace(const Ciphertext<DCRTPoly>& cipher, int d) {
        Plaintext ptx;
        m_cc->Decrypt(m_keyPair.secretKey, cipher, &ptx);
        auto vec = ptx->GetRealPackedValue();
        double tr = 0.0;
        for (int i = 0; i < d; i++) {
            tr += vec[i * d + i];
        }
        return tr;
    }
};
