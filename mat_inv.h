#pragma once

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include <functional>
#include <omp.h>
#include "matrix_inversion_algo.h"
#include "matrix_algo_singlePack.h"
#include "encryption.h"

using namespace lbcrypto;

template <int d> 
struct MatInvContext {
    CryptoContext<DCRTPoly> m_cc;
    PublicKey<DCRTPoly> m_PublicKey;
    Ciphertext<DCRTPoly> input_matrix;
    Ciphertext<DCRTPoly> output_matrix;
    std::string m_outputLocation;

    MatInvContext(std::string ccLocation, std::string pubKeyLocation,
                std::string multKeyLocation, std::string rotKeyLocation,
                std::string matrixLocation, std::string outputLocation)
        : m_outputLocation(outputLocation) {
        initCC(ccLocation, pubKeyLocation, multKeyLocation, rotKeyLocation,
               matrixLocation);
    }

    void initCC(std::string ccLocation, std::string pubKeyLocation,
                std::string multKeyLocation, std::string rotKeyLocation,
                std::string matrixLocation) {
        if (!Serial::DeserializeFromFile(ccLocation, m_cc, SerType::BINARY)) {
            std::cerr << "Could not deserialize cryptocontext file" << std::endl;
            std::exit(1);
        }

        if (!Serial::DeserializeFromFile(pubKeyLocation, m_PublicKey,
                                         SerType::BINARY)) {
            std::cerr << "Could not deserialize public key file" << std::endl;
            std::exit(1);
        }

        std::ifstream multKeyIStream(multKeyLocation,
                                     std::ios::in | std::ios::binary);
        if (!multKeyIStream.is_open()) {
            std::cerr << "Mult key stream not open" << std::endl;
            std::exit(1);
        }
        if (!m_cc->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
            std::cerr << "Could not deserialize mult key file" << std::endl;
            std::exit(1);
        }

        std::ifstream rotKeyIStream(rotKeyLocation,
                                    std::ios::in | std::ios::binary);
        if (!rotKeyIStream.is_open()) {
            std::cerr << "Rot key stream not open" << std::endl;
            std::exit(1);
        }
        if (!m_cc->DeserializeEvalAutomorphismKey(rotKeyIStream,
                                                  SerType::BINARY)) {
            std::cerr << "Could not deserialize eval rot key file" << std::endl;
            std::exit(1);
        }

        if (!Serial::DeserializeFromFile(matrixLocation, input_matrix,
                                         SerType::BINARY)) {
            std::cerr << "Could not deserialize matrix cipher" << std::endl;
            std::exit(1);
        }
    }

    void eval() {
        auto enc = std::make_shared<Encryption>(m_cc, m_PublicKey);
        auto inverter = std::make_unique<MatrixInverse_newColOpt<d>>(enc, m_cc, m_PublicKey);
        output_matrix = inverter->eval_inverse(input_matrix);
    }

    void deserializeOutput() {
        if (!Serial::SerializeToFile(m_outputLocation, output_matrix,
                                     SerType::BINARY)) {
            std::cerr << "Error writing ciphertext" << std::endl;
            std::exit(1);
        }
    }
};