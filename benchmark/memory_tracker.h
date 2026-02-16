#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <openfhe.h>

// Required for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

using namespace lbcrypto;

// ============================================================
// Memory Tracking Utilities
// ============================================================

class MemoryMonitor {
private:
    std::atomic<bool> running_;
    std::thread monitor_thread_;
    std::atomic<double> peak_memory_gb_;
    std::vector<double> samples_gb_;
    mutable std::mutex samples_mutex_;
    int sample_interval_ms_;

    void monitorLoop();
    static double getCurrentMemoryGB();

public:
    explicit MemoryMonitor(int sample_interval_ms = 100);
    ~MemoryMonitor();

    double getPeakMemoryGB() const;
    double getAverageMemoryGB() const;

    // Static helper for one-time snapshot
    static double getMemoryUsageGB();

    // Delete copy/move
    MemoryMonitor(const MemoryMonitor&) = delete;
    MemoryMonitor& operator=(const MemoryMonitor&) = delete;
    MemoryMonitor(MemoryMonitor&&) = delete;
    MemoryMonitor& operator=(MemoryMonitor&&) = delete;
};

// ============================================================
// Serialization Size Utilities
// ============================================================

namespace MemoryUtils {

// Get serialized size of a ciphertext in bytes
inline size_t getCiphertextSize(const Ciphertext<DCRTPoly>& ct) {
    std::stringstream ss;
    Serial::Serialize(ct, ss, SerType::BINARY);
    return ss.str().size();
}

// Get serialized size of rotation keys (EvalAutomorphismKeyMap)
inline size_t getRotationKeysSize(CryptoContext<DCRTPoly>& cc) {
    std::stringstream ss;
    // EvalAutomorphismKeys are stored internally (use static method with default keyTag)
    if (!CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(ss, SerType::BINARY)) {
        return 0;
    }
    return ss.str().size();
}

// Get serialized size of relinearization key
inline size_t getRelinKeySize(CryptoContext<DCRTPoly>& cc) {
    std::stringstream ss;
    if (!cc->SerializeEvalMultKey(ss, SerType::BINARY)) {
        return 0;
    }
    return ss.str().size();
}

// Convert bytes to MB
inline double bytesToMB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// Convert bytes to GB
inline double bytesToGB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Memory tracking results structure
struct MemoryMetrics {
    double idleMemoryGB = 0.0;       // Memory at program start
    double setupMemoryGB = 0.0;      // Memory after CryptoContext + keys
    double peakMemoryGB = 0.0;       // Peak memory during computation
    double avgMemoryGB = 0.0;        // Average memory during computation

    // Serialized sizes
    double ciphertextSizeMB = 0.0;   // Single ciphertext size
    int numCiphertexts = 1;          // Number of ciphertexts used
    double rotationKeysSizeMB = 0.0; // Rotation keys total size
    double relinKeySizeMB = 0.0;     // Relinearization key size

    // Computed overheads
    double setupOverheadGB() const { return setupMemoryGB - idleMemoryGB; }
    double computeOverheadGB() const { return peakMemoryGB - setupMemoryGB; }
    double totalCiphertextsMB() const { return ciphertextSizeMB * numCiphertexts; }

    void print(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(4);
        os << "\n=== Memory Analysis ===" << std::endl;
        os << "  Idle Memory:        " << idleMemoryGB << " GB" << std::endl;
        os << "  Setup Memory:       " << setupMemoryGB << " GB" << std::endl;
        os << "  Peak Memory:        " << peakMemoryGB << " GB" << std::endl;
        os << "  Avg Memory:         " << avgMemoryGB << " GB" << std::endl;
        os << "  Setup Overhead:     " << setupOverheadGB() << " GB" << std::endl;
        os << "  Compute Overhead:   " << computeOverheadGB() << " GB" << std::endl;

        os << "\n=== Serialized Sizes ===" << std::endl;
        os << std::setprecision(2);
        os << "  Ciphertext:         " << ciphertextSizeMB << " MB";
        if (numCiphertexts > 1) {
            os << " x " << numCiphertexts << " = " << totalCiphertextsMB() << " MB";
        }
        os << std::endl;
        if (rotationKeysSizeMB > 0) {
            os << "  Rotation Keys:      " << rotationKeysSizeMB << " MB" << std::endl;
        }
        os << "  Relin Key:          " << relinKeySizeMB << " MB" << std::endl;
    }

    void printCompact(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(3);
        os << "  Memory: setup=" << setupOverheadGB() << "GB, peak=" << computeOverheadGB() << "GB";
        os << ", ct=" << std::setprecision(1) << ciphertextSizeMB << "MB";
        if (numCiphertexts > 1) {
            os << "x" << numCiphertexts;
        }
        if (rotationKeysSizeMB > 0) {
            os << ", rotKey=" << rotationKeysSizeMB << "MB";
        }
        os << std::endl;
    }
};

} // namespace MemoryUtils
