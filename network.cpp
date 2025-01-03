#include "network.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <ctime>
#include <algorithm>

// Perceptron Methods

Perceptron::Perceptron(int inputs, double bias) : bias(bias) {
    weights.resize(inputs + 1); // +1 for the bias
    m.resize(inputs + 1, 0.0);
    v.resize(inputs + 1, 0.0);

    // Initialize weights randomly
    srand(time(NULL));
    std::generate(weights.begin(), weights.end(), []() { return (double)(rand() % 100) / 1000; });
}

double Perceptron::run(const std::vector<double>& x) {
    std::vector<double> input_with_bias = x;
    input_with_bias.push_back(bias);

    // Compute dot product
    return sigmoid(std::inner_product(input_with_bias.begin(), input_with_bias.end(), weights.begin(), 0.0));
}

void Perceptron::set_weights(const std::vector<double>& w_init) {
    weights = w_init;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// MultiLayerPerceptron Methods

MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int>& layers, double bias, bool adam, double eta)
    : layers(layers), bias(bias), eta(eta), use_adam(adam), back_iter(0) {
    srand(time(NULL));

    for (size_t i = 0; i < layers.size(); ++i) {
        values.emplace_back(layers[i], 0.0);
        gradients.emplace_back(layers[i], 0.0);
        loss_gradient.emplace_back(layers[i], 0.0);
        network.emplace_back();
        if (i > 0) {
            for (int j = 0; j < layers[i]; ++j) {
                network[i].emplace_back(Perceptron(layers[i - 1], bias));
            }
        }
    }
}

std::vector<double> MultiLayerPerceptron::run(const std::vector<double>& x) {
    values[0] = x;

    for (size_t i = 1; i < network.size(); ++i) {
        for (size_t j = 0; j < network[i].size(); ++j) {
            values[i][j] = network[i][j].run(values[i - 1]);
        }
    }
    return values.back();
}

std::vector<double> MultiLayerPerceptron::bp(const std::vector<double>& error) {
    back_iter++;
    auto& output_errors = values.back();

    // Calculate gradients for the output layer
    for (size_t i = 0; i < output_errors.size(); ++i) {
        gradients.back()[i] = output_errors[i] * (1 - output_errors[i]) * error[i];
    }

    // Backpropagate errors through the network
    for (int i = (int)network.size() - 2; i > 0; --i) {
        for (size_t j = 0; j < network[i].size(); ++j) {
            double forward_error = 0.0;
            for (size_t k = 0; k < layers[i + 1]; ++k) {
                forward_error += network[i + 1][k].weights[j] * gradients[i + 1][k];
            }
            gradients[i][j] = values[i][j] * (1 - values[i][j]) * forward_error;
        }
    }

    return gradients[1];
}

void MultiLayerPerceptron::set_weights(const std::vector<std::vector<std::vector<double>>>& w_init) {
    for (size_t i = 0; i < w_init.size(); ++i) {
        for (size_t j = 0; j < w_init[i].size(); ++j) {
            network[i + 1][j].set_weights(w_init[i][j]);
        }
    }
}

void MultiLayerPerceptron::print_weights() {
    for (size_t i = 1; i < network.size(); ++i) {
        for (size_t j = 0; j < network[i].size(); ++j) {
            std::cout << "Layer " << i << " Neuron " << j << ": ";
            for (double w : network[i][j].weights) {
                std::cout << w << " ";
            }
            std::cout << std::endl;
        }
    }
}

