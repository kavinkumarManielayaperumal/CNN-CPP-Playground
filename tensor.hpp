// Updated tensor.hpp
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <iostream>

// Tensor class for 4D tensors
class Tensor {
private:
    std::vector<float> data;            // Stores tensor data
    std::vector<int> shape;             // Shape of the tensor (n, c, h, w)
    int size;                           // Total number of elements in the tensor

public:
    // Constructor: Initialize tensor with the given shape and set all elements to zero
    Tensor(const std::vector<int>& shape) : shape(shape) {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        data.resize(size, 0.0f); // Initialize with zeros
    }

    // Accessor for tensor elements (4D indexing)
    float& at(int n, int c, int h, int w) {
        int index = n * shape[1] * shape[2] * shape[3] +
                    c * shape[2] * shape[3] +
                    h * shape[3] +
                    w;
        if (index >= size || index < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // Const accessor
    const float& at(int n, int c, int h, int w) const {
        int index = n * shape[1] * shape[2] * shape[3] +
                    c * shape[2] * shape[3] +
                    h * shape[3] +
                    w;
        if (index >= size || index < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    // Getter for raw data
    std::vector<float>& getData() { return data; }

    // Const getter for raw data
    const std::vector<float>& getData() const { return data; }

    // Get the shape of the tensor
    const std::vector<int>& getShape() const { return shape; }

    // Reshape the tensor
    void reshape(const std::vector<int>& new_shape) {
        int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        if (new_size != size) {
            throw std::invalid_argument("New shape size must match original size");
        }
        shape = new_shape;
    }
};

#endif // TENSOR_HPP