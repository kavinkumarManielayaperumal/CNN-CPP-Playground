#include "tensor.hpp"
#include "network.hpp"
#include <iostream>

int main() {
    try {
        // Input tensor: Batch size 1, 1 channel, 5x5 image
        Tensor input({1, 1, 5, 5});
        for (int i = 0; i < 25; ++i) {
            input.getData().at(i) = i + 1; // Fill with sequential values
        }

        // Create Conv2D layer: 1 output channel, 1 input channel, 3x3 kernel
        Conv2D conv2d(1, 1, 3, 3);

        // Fill weights and biases
        for (int i = 0; i < 9; ++i) { // Fill kernel with 1s
            conv2d.getWeights().getData().at(i) = 1;
        }
        conv2d.getBiases().getData().at(0) = 0; // Set bias to 0

        // Perform forward pass
        Tensor output = conv2d.fwd(input);

        // Print output
        const auto& output_shape = output.getShape();
        std::cout << "Output shape: ";
        for (int dim : output_shape) std::cout << dim << " ";
        std::cout << std::endl;

        for (int h = 0; h < output_shape[2]; ++h) {
            for (int w = 0; w < output_shape[3]; ++w) {
                std::cout << output.getData().at(h * output_shape[3] + w) << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
