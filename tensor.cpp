#include "tensor.hpp"

// Default constructor
Tensor::Tensor() {}

// 2D tensor
Tensor::Tensor(int H, int W) : _dim(2), _length(H * W) {
    _data.resize(_length, 0);
    _shape = {H, W};
}

// 3D tensor
Tensor::Tensor(int H, int W, int Depth) : _dim(3), _length(H * W * Depth) {
    _data.resize(_length, 0);
    _shape = {H, W, Depth};
}

// 4D tensor
Tensor::Tensor(int Layers, int H, int W, int Depth) : _dim(4), _length(Layers * H * W * Depth) {
    _data.resize(_length, 0);
    _shape = {Layers, H, W, Depth};
}

// Tensor with custom dimensions
Tensor::Tensor(int* shapes, int dimensions) : _dim(dimensions) {
    _length = 1;
    _shape.assign(shapes, shapes + dimensions);
    for (int i = 0; i < dimensions; ++i) _length *= shapes[i];
    _data.resize(_length, 0);
}

// Initialize tensor
void Tensor::init(const int* shapes, int dimensions) {
    if (!_data.empty()) {
        cerr << "Error: Tensor already initialized." << endl;
        return;
    }

    _dim = dimensions;
    _shape.assign(shapes, shapes + dimensions);
    _length = 1;
    for (int i = 0; i < dimensions; ++i) _length *= shapes[i];
    _data.resize(_length, 0);
}

// Rebuild tensor
void Tensor::rebuild(int* shapes, int dimensions) {
    _dim = dimensions;
    _shape.assign(shapes, shapes + dimensions);
    _length = 1;
    for (int i = 0; i < dimensions; ++i) _length *= shapes[i];
    _data.assign(_length, 0);
}

// Get specific dimension size
int Tensor::get_shape(int dim_n) const {
    return _shape[dim_n];
}

// Get total number of elements
int Tensor::get_length() const {
    return _length;
}

// Access value at a specific index
double Tensor::get_value(int* index, int dimensions) const {
    if (dimensions != _dim) {
        cerr << "Error: Dimension mismatch." << endl;
        return -1;
    }

    int offset = 0;
    for (int i = 0; i < dimensions; ++i) {
        int factor = 1;
        for (int j = 0; j < i; ++j) factor *= _shape[j];
        offset += index[i] * factor;
    }
    return _data[offset];
}

// Get tensor data
vector<double>& Tensor::get_data() {
    return _data;
}

// Assign value at specific index
void Tensor::assign(double val, int* index, int dimensions) {
    if (dimensions != _dim) {
        cerr << "Error: Dimension mismatch." << endl;
        return;
    }

    int offset = 0;
    for (int i = 0; i < dimensions; ++i) {
        int factor = 1;
        for (int j = 0; j < i; ++j) factor *= _shape[j];
        offset += index[i] * factor;
    }
    _data[offset] = val;
}

// Add value at specific index
void Tensor::add(double val, int* index, int dimensions) {
    if (dimensions != _dim) {
        cerr << "Error: Dimension mismatch." << endl;
        return;
    }

    int offset = 0;
    for (int i = 0; i < dimensions; ++i) {
        int factor = 1;
        for (int j = 0; j < i; ++j) factor *= _shape[j];
        offset += index[i] * factor;
    }
    _data[offset] += val;
}

// Assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    _data = other._data;
    _shape = other._shape;
    _dim = other._dim;
    _length = other._length;
    return *this;
}

// Access operator for flat data
double& Tensor::operator[](int index) {
    if (index >= _length) {
        cerr << "Index out of bounds. Returning last element." << endl;
        return _data.back();
    }
    return _data[index];
}
