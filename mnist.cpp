#include "mnist.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>

// Constructor
MNISTLoader::MNISTLoader(const std::string& images_path, const std::string& labels_path)
    : imagesPath(images_path), labelsPath(labels_path) {}

// Load the dataset
void MNISTLoader::loadDataset() {
    loadImages();
    loadLabels();
}

// Load images from the IDX file
void MNISTLoader::loadImages() {
    std::ifstream file(imagesPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open images file: " + imagesPath);
    }

    int32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic != 2051) {
        throw std::runtime_error("Invalid magic number for images file");
    }

    images = Tensor({num_images, rows, cols});
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                images.at(i, r, c) = static_cast<float>(pixel) / 255.0f; // Normalize
            }
        }
    }
    file.close();
}

// Load labels from the IDX file
void MNISTLoader::loadLabels() {
    std::ifstream file(labelsPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labelsPath);
    }

    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (magic != 2049) {
        throw std::runtime_error("Invalid magic number for labels file");
    }

    labels = Tensor({num_labels});
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.at(i) = static_cast<float>(label);
    }
    file.close();
}

// Get images
Tensor MNISTLoader::getImages() const {
    return images;
}

// Get labels
Tensor MNISTLoader::getLabels() const {
    return labels;
}
