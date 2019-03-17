#include <iostream>
#include <vector>
#include <algorithm>
#include "mpi.h"

double RandomDouble(double min = -1, double max = 1)
{
    double f = (double)std::rand() / RAND_MAX;
    return min + f * (max - min);
}

std::vector<double> RandomDoubleVector(size_t size, double min = -1, double max = 1)
{
    auto vector = std::vector<double>(size);
    for (int i = 0; i < size; ++i) {
        vector[i] = RandomDouble(min, max);
    }

    return vector;
}

class Matrix {
public:
    Matrix(size_t size) : _size(size), _data(size * size) {}
    Matrix(size_t size, std::vector<double> data) : _size(size), _data(size * size) {
        _data = data;
    }
    double& operator()(size_t i, size_t j) {
        return _data[i * _size + j];
    }
    double operator()(size_t i, size_t j) const {
        return _data[i * _size + j];
    }
    static Matrix I(size_t size) {
        auto matrix = Matrix(size);
        for (size_t i = 0; i < size; ++i)
            matrix(i, i) = 1;

        return matrix;
    }
    static Matrix Zero(size_t size) {
        auto matrix = Matrix(size);
        return matrix;
    }

    static Matrix RandomMatrix(size_t size, double min = -1, double max = 1) {
        auto matrix = Matrix(size);
        for (size_t i = 0; i < size; ++i)
            for (size_t j = 0; j < size; ++j)
                matrix(i, j) = RandomDouble(min, max);

        return matrix;
    }

    Matrix operator+(const Matrix& b) {
        auto sum = Matrix(_size, _data);
        for (size_t i = 0; i < _size; ++i)
            for (size_t j = 0; j < _size; ++j)
                sum(i, j) += b(i, j);

        return sum;
    }

    Matrix operator+(const double & d) {
        auto sum = Matrix(_size, _data);
        for (size_t i = 0; i < _size; ++i)
            sum(i, i) += d;

        return sum;
    }

    Matrix operator+=(const Matrix& b) {
        for (size_t i = 0; i < _size; ++i)
            for (size_t j = 0; j < _size; ++j)
                operator()(i, j) += b(i, j);

        return *this;
    }

    Matrix operator*(const Matrix& b) {
        auto product = Matrix(_size);

        for (size_t i = 0; i < _size; ++i)
            for (size_t j = 0; j < _size; ++j)
                for (size_t k = 0; k < _size; ++k) {
                    product(i, j) += operator()(i, k) * b(k, j);
                }

        return product;
    }

    Matrix operator*(const double& d) {
        auto scalarMult = Matrix(_size, _data);
        for (size_t i = 0; i < _size; ++i)
            for (size_t j = 0; j < _size; ++j)
                scalarMult(i, j) *= d;

        return scalarMult;
    }

    Matrix Pow(size_t n) {
        auto result = Matrix(_size);

        for (size_t i = 0; i < _size; i++)
        {
            result(i, i) = 1;
        }

        for (size_t i = 0; i < n; i++) {
            result = result * *this;
        }

        return result;
    }

    void Show() {
        for (size_t i = 0; i < _size; ++i) {
            for (size_t j = 0; j < _size; ++j)
                std::cout << operator()(i, j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    size_t GetSize() {
        return _size;
    }
private:
    size_t _size;
    std::vector<double> _data;
};

Matrix ForwardSum(std::vector<double> polinomialCoefs, Matrix& A)
{
    auto result = Matrix(A.GetSize());
    for (size_t n = 0; n < polinomialCoefs.size(); n++) {
        result += A.Pow(n) * polinomialCoefs[n];
    }

    return result;
}

Matrix Horner(std::vector<double> polinomialCoefs, Matrix& A) {
    size_t matrixSize = A.GetSize();

    auto result = Matrix::I(matrixSize) * polinomialCoefs[polinomialCoefs.size() - 1];
    for (int n = polinomialCoefs.size() - 2; n >= 0; n--) {
        result = result * A + Matrix::I(matrixSize) * polinomialCoefs[n];
    }

    return result;
}

Matrix SimpleParallelHorner(std::vector<double> polinomialCoefs, Matrix &A, int threadsCount) {
    size_t matrixSize = A.GetSize();

    auto result = Matrix::Zero(A.GetSize());

    for (int k = 0; k < threadsCount; k++)
    {
        auto part = Matrix::Zero(matrixSize);
        for (int i = k, j = 0; i < polinomialCoefs.size(); i += threadsCount, j += threadsCount) {
            part += A.Pow(j) * polinomialCoefs[i];
        }

        part = A.Pow(k) * part;

        result += part;
    }

    return result;
}

Matrix TrueParallelHorner(std::vector<double> polinomialCoefs, Matrix &A, int threadsCount) {
    size_t matrixSize = A.GetSize();

    auto result = Matrix::Zero(A.GetSize());

    for (int k = 0; k < threadsCount; k++)
    {
        std::vector<size_t> indexes(0);
        for (int i = k; i < polinomialCoefs.size(); i += threadsCount) {
            indexes.push_back(i);
        }
        std::reverse(std::begin(indexes), std::end(indexes));

		auto APowered = A.Pow(threadsCount);
		auto I = Matrix::I(matrixSize);
		auto part = Matrix::I(matrixSize) * polinomialCoefs[indexes[0]];
        for (int n = 1; n < indexes.size(); n++) {
            part = part * APowered + I * polinomialCoefs[indexes[n]];
        }

        part = A.Pow(k) * part;

        result += part;
    }

    return result;
}

int main() {
    size_t matrixSize = 200;
    auto A = Matrix::RandomMatrix(matrixSize);
    std::vector<double> polinomialCoefs = RandomDoubleVector(20, -5, 5);

    size_t threadsCount = 1;

    std::cout << "Improved parallel Horner method:" << std::endl;
	std::cout << "Matrix size: " << matrixSize << "x" << matrixSize << std::endl;
	std::cout << "Polynomial rang: " << polinomialCoefs.size() << std::endl;
	std::cout << "Threads: " << threadsCount << std::endl;

    clock_t begin = clock();
    TrueParallelHorner(polinomialCoefs, A, threadsCount);
    clock_t end = clock();
    std::cout << "Done in " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl;

    return 0;
}