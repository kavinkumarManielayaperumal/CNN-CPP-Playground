#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <ctime>

using namespace std;

// Define constants for Adam optimizer
#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-7
#define ALPHA 0.001

// Activation Functions
class ReLU {
public:
    static vector<double> apply(const vector<double>& input) {
        vector<double> output(input.size());
        transform(input.begin(), input.end(), output.begin(),
                  [](double x) { return max(0.0, x); });
        return output;
    }

    static vector<double> derivative(const vector<double>& input) {
        vector<double> output(input.size());
        transform(input.begin(), input.end(), output.begin(),
                  [](double x) { return x > 0 ? 1.0 : 0.0; });
        return output;
    }
};

// Perceptron Class
class Perceptron {
public:
    Perceptron(int inputs, double bias = 1.0);
    double run(const vector<double>& x);
    void set_weights(const vector<double>& w_init);
    double sigmoid(double x);

private:
    vector<double> weights;
    vector<double> m, v;
    double bias;
};

// Perceptron Implementation
Perceptron::Perceptron(int inputs, double bias) : bias(bias) {
    weights.resize(inputs + 1); // +1 for the bias
    m.resize(inputs + 1, 0.0);
    v.resize(inputs + 1, 0.0);
    generate(weights.begin(), weights.end(), []() { return (double)(rand() % 100) / 1000; });
}

double Perceptron::run(const vector<double>& x) {
    vector<double> input_with_bias = x;
    input_with_bias.push_back(bias);
    return sigmoid(inner_product(input_with_bias.begin(), input_with_bias.end(), weights.begin(), 0.0));
}

void Perceptron::set_weights(const vector<double>& w_init) {
    weights = w_init;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Multi-Layer Perceptron (MLP)
class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(const vector<int>& layers, double bias = 1.0, bool adam = true, double eta = 0.01);
    vector<double> run(const vector<double>& x);
    vector<double> bp(const vector<double>& error);
    void set_weights(const vector<vector<vector<double>>>& w_init);
    void print_weights();

private:
    vector<int> layers;
    vector<vector<Perceptron>> network;
    vector<vector<double>> values;
    vector<vector<double>> gradients;
    vector<vector<double>> loss_gradient;
    double bias;
    double eta;
    bool use_adam;
    int back_iter;

    double Adam(double& m, double& v, double derivative);
};

// MLP Implementation
MultiLayerPerceptron::MultiLayerPerceptron(const vector<int>& layers, double bias, bool adam, double eta)
    : layers(layers), bias(bias), eta(eta), use_adam(adam), back_iter(0) {
    srand(time(NULL));

    for (size_t i = 0; i < layers.size(); ++i) {
        values.push_back(vector<double>(layers[i], 0.0));
        gradients.push_back(vector<double>(layers[i], 0.0));
        loss_gradient.push_back(vector<double>(layers[i], 0.0));
        network.push_back(vector<Perceptron>());
        if (i > 0) {
            for (int j = 0; j < layers[i]; ++j) {
                network[i].push_back(Perceptron(layers[i - 1], bias));
            }
        }
    }
}

vector<double> MultiLayerPerceptron::run(const vector<double>& x) {
    values[0] = x;
    for (size_t i = 1; i < network.size(); ++i) {
        for (size_t j = 0; j < network[i].size(); ++j) {
            values[i][j] = network[i][j].run(values[i - 1]);
        }
    }
    return values.back();
}

vector<double> MultiLayerPerceptron::bp(const vector<double>& error) {
    vector<double> output_errors = values.back();
    back_iter++;

    // Output layer gradients
    for (size_t i = 0; i < output_errors.size(); ++i) {
        gradients.back()[i] = output_errors[i] * (1 - output_errors[i]) * error[i];
    }

    // Backpropagate
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

void MultiLayerPerceptron::set_weights(const vector<vector<vector<double>>>& w_init) {
    for (size_t i = 0; i < w_init.size(); ++i) {
        for (size_t j = 0; j < w_init[i].size(); ++j) {
            network[i + 1][j].set_weights(w_init[i][j]);
        }
    }
}

void MultiLayerPerceptron::print_weights() {
    for (size_t i = 1; i < network.size(); ++i) {
        for (size_t j = 0; j < network[i].size(); ++j) {
            cout << "Layer " << i << " Neuron " << j << ": ";
            for (double w : network[i][j].weights) {
                cout << w << " ";
            }
            cout << endl;
        }
    }
}

#endif
