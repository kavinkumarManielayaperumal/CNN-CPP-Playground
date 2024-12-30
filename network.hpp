#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"
#include <vector>
#include <stdexcept>

// Conv2D Layer Class
class Conv2D {
private:
    Tensor weights; // Shape: (C_out, C_in, K_h, K_w)
    Tensor biases;  // Shape: (C_out)
    int stride;
    int padding;

public:
    // Constructor
    Conv2D(int C_out, int C_in, int K_h, int K_w, int stride = 1, int padding = 0)
        : weights({C_out, C_in, K_h, K_w}), biases({C_out}), stride(stride), padding(padding) {}

    // Forward pass for Conv2D
    Tensor fwd(const Tensor& input) {
        // Get input and kernel dimensions
        const auto& input_shape = input.getShape();
        int N = input_shape[0];
        int C_in = input_shape[1];
        int H_in = input_shape[2];
        int W_in = input_shape[3];

        const auto& kernel_shape = weights.getShape();
        int C_out = kernel_shape[0];
        int K_h = kernel_shape[2];
        int K_w = kernel_shape[3];

        // Calculate output dimensions
        int H_out = (H_in - K_h + 2 * padding) / stride + 1;
        int W_out = (W_in - K_w + 2 * padding) / stride + 1;

        // Initialize output tensor
        Tensor output({N, C_out, H_out, W_out});

        // Perform convolution
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < C_out; ++co) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float value = biases.getData().at(co); // Start with bias
                        for (int ci = 0; ci < C_in; ++ci) {
                            for (int kh = 0; kh < K_h; ++kh) {
                                for (int kw = 0; kw < K_w; ++kw) {
                                    int h_in = h * stride + kh - padding;
                                    int w_in = w * stride + kw - padding;
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        value += input.getData().at(n * C_in * H_in * W_in +
                                                                    ci * H_in * W_in +
                                                                    h_in * W_in + w_in) *
                                                 weights.getData().at(co * C_in * K_h * K_w +
                                                                      ci * K_h * K_w +
                                                                      kh * K_w + kw);
                                    }
                                }
                            }
                        }
                        output.getData().at(n * C_out * H_out * W_out +
                                            co * H_out * W_out +
                                            h * W_out + w) = value;
                    }
                }
            }
        }

        return output;
    }

    // Access weights and biases
    Tensor& getWeights() { return weights; }
    Tensor& getBiases() { return biases; }
};

#endif // NETWORK_HPP
