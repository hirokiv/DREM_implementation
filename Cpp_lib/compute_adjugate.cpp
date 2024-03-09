#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// #include <Eigen/Core>
#include <Eigen/Dense>
#include <stdexcept> // For std::runtime_error
// #include <eigen3/Eigen/Dense>


Eigen::MatrixXd computeAdjugate(const Eigen::MatrixXd& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::runtime_error("Adjugate is only defined for square matrices.");
    }

    Eigen::MatrixXd adjugateMatrix(matrix.rows(), matrix.cols());

    // Iterate over each element in the matrix to compute the cofactor matrix
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            // Compute the submatrix by removing the i-th row and j-th column
            Eigen::MatrixXd submatrix(matrix.rows() - 1, matrix.cols() - 1);
            
            // Fill the submatrix
            for (int sub_i = 0; sub_i < matrix.rows(); ++sub_i) {
                for (int sub_j = 0; sub_j < matrix.cols(); ++sub_j) {
                    if (sub_i != i && sub_j != j) {
                        int dest_i = sub_i < i ? sub_i : sub_i - 1;
                        int dest_j = sub_j < j ? sub_j : sub_j - 1;
                        submatrix(dest_i, dest_j) = matrix(sub_i, sub_j);
                    }
                }
            }

            // Compute the cofactor
            double cofactor = std::pow(-1, i + j) * submatrix.determinant();

            // Assign the cofactor to the transposed position to form the adjugate
            adjugateMatrix(j, i) = cofactor;
        }
    }

    return adjugateMatrix;
}


PYBIND11_MODULE(adjugate_module, m) {
    m.def("compute_adjugate", &computeAdjugate, "A function that computes the adjugate of a matrix");
}

