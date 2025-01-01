#include "mnist.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>

std::vector<std::vector<uint8_t>> MNISTLoader::loadImages(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    // Read metadata
    int32_t magicNumber, numImages, numRows, numCols;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    // Convert from big-endian
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    // Load image data
    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));
    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);
    }

    return images;
}

std::vector<uint8_t> MNISTLoader::loadLabels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    // Read metadata
    int32_t magicNumber, numLabels;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numLabels), 4);

    // Convert from big-endian
    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    // Load label data
    std::vector<uint8_t> labels(numLabels);
    file.read(reinterpret_cast<char*>(labels.data()), numLabels);

    return labels;
}
