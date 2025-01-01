#pragma once
#include <vector>
#include <string>

class MNISTLoader {
public:
    // Methods to load data
    std::vector<std::vector<uint8_t>> loadImages(const std::string& filePath);
    std::vector<uint8_t> loadLabels(const std::string& filePath);
};
