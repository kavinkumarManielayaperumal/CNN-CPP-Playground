#include "mnist.hpp"

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void normalize_set(Tensor& set, int len, int n_rows, int n_cols) {
    for (int img = 0; img < len; ++img) {
        double max_val = 0, min_val = 255, val;

        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                int index[4] = {img, 0, r, c};
                val = set.get_value(index, 4);
                if (val > max_val) max_val = val;
                if (val < min_val) min_val = val;
            }
        }

        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                int index[4] = {img, 0, r, c};
                val = set.get_value(index, 4);
                val = (val - min_val) / (max_val - min_val);
                set.assign(val, index, 4);
            }
        }
    }
}

void MNIST::get_set(string path, int num_images, Tensor& set) {
    ifstream file(path, ios::binary);
    if (file.is_open()) {
        int magic_number = 0, num_rows = 0, num_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&num_images, sizeof(num_images));
        num_images = ReverseInt(num_images);
        file.read((char*)&num_rows, sizeof(num_rows));
        num_rows = ReverseInt(num_rows);
        file.read((char*)&num_cols, sizeof(num_cols));
        num_cols = ReverseInt(num_cols);

        for (int i = 0; i < num_images; ++i) {
            for (int r = 0; r < num_rows; ++r) {
                for (int c = 0; c < num_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    int index[4] = {i, 0, r, c};
                    set.assign((double)temp, index, 4);
                }
            }
        }
        file.close();
    }
}

void MNIST::get_label(string path, int num_images, vector<int>& labels) {
    ifstream file(path, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&num_images, sizeof(num_images));
        num_images = ReverseInt(num_images);

        for (int i = 0; i < num_images; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels[i] = (int)temp;
        }
        file.close();
    }
}

void MNIST::init_mnist(Tensor& train_ds, vector<int>& train_labels,
                       Tensor& test_ds, vector<int>& test_labels) {
    int train_shapes[4] = {MNIST_TRAIN_SHAPES};
    int test_shapes[4] = {MNIST_TEST_SHAPES};

    train_ds.init(train_shapes, 4);
    test_ds.init(test_shapes, 4);
    train_labels.assign(MNIST_TRAIN_LEN, 0);
    test_labels.assign(MNIST_TEST_LEN, 0);
}

void MNIST::load_mnist(Tensor& train_ds, vector<int>& train_labels,
                       Tensor& test_ds, vector<int>& test_labels) {
    init_mnist(train_ds, train_labels, test_ds, test_labels);
    cout << "Loading MNIST dataset..." << endl;

    get_set("MNIST_data/train-images.idx3-ubyte", MNIST_TRAIN_LEN, train_ds);
    get_label("MNIST_data/train-labels.idx1-ubyte", MNIST_TRAIN_LEN, train_labels);

    get_set("MNIST_data/t10k-images.idx3-ubyte", MNIST_TEST_LEN, test_ds);
    get_label("MNIST_data/t10k-labels.idx1-ubyte", MNIST_TEST_LEN, test_labels);

    normalize_set(train_ds, MNIST_TRAIN_LEN, 28, 28);
    normalize_set(test_ds, MNIST_TEST_LEN, 28, 28);
}
