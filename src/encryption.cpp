#include "encryption.h"
#include "ciphertext-fwd.h"
#include "lattice/hal/lat-backend.h"

Ciphertext<DCRTPoly> Encryption::encryptInput(std::vector<double> input) {
    assert(input.size() <= m_cc->GetEncodingParams()->GetBatchSize() &&
           "Input size should not larger than Batch Size");

    if (!this->m_cc) {
        throw std::runtime_error("CryptoContext is not initialized");
    }

    if (!this->m_PublicKey) {
        throw std::runtime_error("Public key is not initialized");
    }
    Plaintext plaintext = m_cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, input.size());
    auto ciphertext = m_cc->Encrypt(m_PublicKey, plaintext);
    return ciphertext;
}

[[nodiscard]] std::vector<double>
DebugEncryption::getPlaintext(const Ciphertext<DCRTPoly> &ct,
                              double threshold) const {
    Plaintext decryptedResult;
    m_cc->Decrypt(m_PrivateKey, ct, &decryptedResult);
    std::vector<double> result = decryptedResult->GetRealPackedValue();

    for (auto &value : result) {
        if (std::abs(value) < threshold) {
            value = 0.0;
        }
    }
    return result;
}

Plaintext DebugEncryption::getDecrypt(const Ciphertext<DCRTPoly> &ct) const {
    Plaintext decryptedResult;
    m_cc->Decrypt(m_PrivateKey, ct, &decryptedResult);
    return decryptedResult;
}
