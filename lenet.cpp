#include <iostream>
#include "tensor.hpp"

int main() {
    try {
        // Step 1: Create a Tensor of shape (1, 3, 4, 4)
        Tensor tensor({1, 3, 4, 4});
        std::cout << "Tensor created successfully with shape: ";
        for (int dim : tensor.getShape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        // Step 2: Assign a value to an element
        tensor(0, 0, 0, 0) = 1.5f;  // Access and modify an element
        std::cout << "Element at (0, 0, 0, 0): " << tensor(0, 0, 0, 0) << std::endl;

        // Step 3: Reshape the Tensor
        tensor.reshape({1, 12, 4});
        std::cout << "Tensor reshaped successfully to: ";
        for (int dim : tensor.getShape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        // Verify element after reshaping
        //std::cout << "Element at new shape index (0, 0, 0): " << tensor(0, 0, 0) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
