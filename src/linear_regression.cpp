#include "ml/linear_regression.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace ml {
LinearRegression::LinearRegression(const Eigen::MatrixXd &training_data,
                                   const Eigen::VectorXd &testing_data,
                                   double learning_rate) {
  this->training_data = training_data;
  this->testing_data = testing_data;
  this->thetas = Eigen::VectorXd::Zero(training_data.cols() + 1);
  this->learning_rate = learning_rate;
}

double LinearRegression::calculate_hypothesis(Eigen::VectorXd thetas,
                                              Eigen::VectorXd row) {
  double res = thetas(0);
  for (Eigen::Index i = 0; i < row.size(); i++) {
    res += row(i) * thetas(i + 1);
  }
  return res;
}
} // namespace ml
