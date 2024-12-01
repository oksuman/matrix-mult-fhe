#include "mat_inv.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

int main(int argc, char *argv[]) {
    std::string ccLocation;
    std::string pubKeyLocation;
    std::string multKeyLocation;
    std::string rotKeyLocation;
    std::string matrixLocation;
    std::string outputLocation;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--cc") {
            ccLocation = argv[i + 1];
        } else if (arg == "--key_pub") {
            pubKeyLocation = argv[i + 1];
        } else if (arg == "--key_mult") {
            multKeyLocation = argv[i + 1];
        } else if (arg == "--key_rot") {
            rotKeyLocation = argv[i + 1];
        } else if (arg == "--input") {
            matrixLocation = argv[i + 1];
        } else if (arg == "--output") {
            outputLocation = argv[i + 1];
        }
    }

    auto procs = omp_get_num_procs();
    omp_set_num_threads(procs / 2);

    MatInvContext<64> context(ccLocation, pubKeyLocation, multKeyLocation,
                           rotKeyLocation, matrixLocation, outputLocation);
    context.eval();
    context.deserializeOutput();

    return 0;
}