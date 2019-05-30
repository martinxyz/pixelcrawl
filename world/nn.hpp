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

// measured 2x overall slowdown when using MatrixXf instead of template
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
    for (int i=0; i<n_hidden; i++) a1(i) = relu(a1(i));  // faster than unaryExpr
    Matrix<float, n_outputs, 1> a2 = w1 * a1 + b1;

    // somewhat hacky residual connection, but it measurably helps
    a2 *= 0.1;
    static_assert(n_outputs <= n_hidden, "n_hidden must be >= n_outputs");
    for (int i=0; i<n_outputs; i++) {
      a2(i) += a1(i);
    }
    a2 *= 1.0 / (1.0 + 0.1);

    return a2;
  }
};
