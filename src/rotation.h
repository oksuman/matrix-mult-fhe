#pragma once
#include <cmath>
#include <openfhe.h>
#include <vector>
#include <set>
#include <string>

using namespace lbcrypto;

class RotationComposer {
private:
    static constexpr int MAX_BATCH = 1 << 15;
    CryptoContext<DCRTPoly> m_cc;
    std::set<int> m_rotationIndices; 

    int getMaxBitPosition() const {
        return static_cast<int>(std::log2(MAX_BATCH));
    }

    std::vector<int> decomposeBinary(int rotation) const {
        std::vector<int> steps;
        const int maxBit = getMaxBitPosition();
        for (int i = maxBit - 1; i >= 0; --i) {
            int stepSize = (1 << i);
            if (stepSize < MAX_BATCH && (rotation & stepSize)) {
                steps.push_back(stepSize);
            }
        }
        return steps;
    }

public:
    RotationComposer(CryptoContext<DCRTPoly> cc)
        : m_cc(cc) {}

    Ciphertext<DCRTPoly> rotate(const Ciphertext<DCRTPoly>& input,
                               int rotation) {
        m_rotationIndices.insert(rotation);

        auto steps = decomposeBinary(std::abs(rotation));
        auto result = input->Clone();
        for (int step : steps) {
            if (rotation > 0) {
                result = m_cc->EvalRotate(result, step);
            } else {
                result = m_cc->EvalRotate(result, -step);
            }
        }
        return result;
    }

    std::vector<int> decomposeForDebug(int rotation) const {
        return decomposeBinary(rotation);
    }

    std::string getRotationIndices() const {
        std::string result = "{";
        bool first = true;
        for (const auto& index : m_rotationIndices) {
            if (!first) {
                result += ", ";
            }
            result += std::to_string(index);
            first = false;
        }
        result += "}";
        return result;
    }

    const std::set<int>& getRotationIndicesSet() const {
        return m_rotationIndices;
    }
};