// simple feed-forward neural network
#pragma once
#include <Eigen/Core>

using Eigen::Matrix;

// MatrixXd Sigmoid(MatrixXd const& x) {
//   return 1.0 / (1.0 + (-x).array().exp());
// }

inline float relu(float x) {
  return (x > 0) ? x : 0;
}

template <int n_inputs, int n_hidden, int n_outputs>
class SmallNN {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Matrix<float, n_hidden, n_inputs> w0;
  Matrix<float, n_hidden, 1> b0;
  Matrix<float, n_outputs, n_hidden> w1;
  Matrix<float, n_outputs, 1> b1;

  Matrix<float, n_outputs, 1> predict(Matrix<float, n_inputs, 1> inputs) {
    Matrix<float, n_hidden, 1> a1 = w0 * inputs + b0;
    a1 = a1.unaryExpr(std::ptr_fun(relu));
    Matrix<float, n_outputs, 1> a2 = w1 * a1 + b1;
    return a2;
  }
};
