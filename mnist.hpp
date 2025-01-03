#ifndef MNIST_HPP
#define MNIST_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include "tensor.hpp"

#define MNIST_TRAIN_LEN 60000
#define MNIST_TEST_LEN 10000
#define MNIST_TRAIN_SHAPES MNIST_TRAIN_LEN, 1, 28, 28
#define MNIST_TEST_SHAPES MNIST_TEST_LEN, 1, 28, 28
#define IMAGE_DATA 784  // 28 x 28

using namespace std;

int ReverseInt(int i);
void normalize_set(Tensor& set, int len, int n_rows, int n_cols);

class MNIST {
private:
    void get_set(string path, int num_images, Tensor& set);
    void get_label(string path, int num_images, vector<int>& labels);
    void init_mnist(Tensor& train_ds, vector<int>& train_labels,
                    Tensor& test_ds, vector<int>& test_labels);

public:
    void load_mnist(Tensor& train_ds, vector<int>& train_labels,
                    Tensor& test_ds, vector<int>& test_labels);
};

#endif
