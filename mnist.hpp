#ifndef MNIST_HPP
#define MNIST_HPP

#include <string>
#include "tensor.hpp"

class MNISTLoader {
public:
    // Constructor
    MNISTLoader(const std::string& images_path, const std::string& labels_path);

    // Load the dataset
    void loadDataset();

    // Get training data
    Tensor getImages() const;
    Tensor getLabels() const;

private:
    std::string imagesPath;
    std::string labelsPath;

    Tensor images; // Tensor to store the image data
    Tensor labels; // Tensor to store the label data

    // Helper methods
    void loadImages();
    void loadLabels();
};

#endif // MNIST_HPP
