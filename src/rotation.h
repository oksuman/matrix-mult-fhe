#pragma once
#include <cmath>
#include <openfhe.h>
#include <vector>
using namespace lbcrypto;

class RotationComposer {
  private:
    static constexpr int MAX_BATCH = 1 << 15; 
    CryptoContext<DCRTPoly> m_cc;

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
        : m_cc(cc){}

    Ciphertext<DCRTPoly> rotate(const Ciphertext<DCRTPoly> &input,
                                int rotation) {
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
};