#include <iostream>
#include <vector>
#include <algorithm>
#include "Matrix.h"

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