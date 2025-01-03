#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <vector>

using namespace std;

class Tensor {
    vector<double> _data;
    vector<int> _shape;
    int _dim = 0, _length = 0;

public:
    Tensor();  // Default constructor

    Tensor(int H, int W);
    Tensor(int H, int W, int Depth);
    Tensor(int Layers, int H, int W, int Depth);
    Tensor(int* shapes, int dimensions);

    void init(const int* shapes, int dimensions);
    void rebuild(int* shapes, int dimensions);

    int get_shape(int dim_n) const;
    int get_length() const;
    double get_value(int* index, int dimensions) const;
    vector<double>& get_data();

    void assign(double val, int* index, int dimensions);
    void add(double val, int* index, int dimensions);

    Tensor& operator=(const Tensor& other);

    double& operator[](int index);
};

#endif
