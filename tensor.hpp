#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>

class Tensor {
private:
    std::vector<float> data;
    std::vector<int> shape;
    int size;

public:
    // Constructor to initialize tensor with given shape
      Tensor(const std::vector<int>& shape) : shape(shape) {
       size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
       data.resize(size, 0.0f); // Initialize with zeros
     }

    // Accessor for tensor elements (4D indexing)
    float& operator()(int n, int c, int h, int w) {
        int index = n * shape[1] * shape[2] * shape[3] +
                    c * shape[2] * shape[3] +
                    h * shape[3] +
                    w;
        if (index >= size || index < 0)
            throw std::out_of_range("Index out of range");
        return data[index];
    }

    // Get the shape of the tensor
    const std::vector<int>& getShape() const {
        return shape;
    }

    // Reshape the tensor
    void reshape(const std::vector<int>& new_shape) {
        int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        if (new_size != size)
            throw std::invalid_argument("New shape size must match original size");
        shape = new_shape;
    }
};

#endif // TENSOR_HPP
