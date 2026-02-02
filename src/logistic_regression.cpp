#include "ml/logistic_regression.hpp"
#include "Eigen/Dense"
#include <cmath>

namespace ml {
LogisticRegression::LogisticRegression(Eigen::MatrixXd X_train,
                                       Eigen::VectorXd Y_train,
                                       Eigen::MatrixXd X_test,
                                       Eigen::VectorXd Y_test,
                                       double learning_rate, double max_iter) {
  this->learning_rate = learning_rate;
  this->max_iter = max_iter;
  this->thetas = Eigen::VectorXd::Zero(static_cast<int>(X_train.cols()) + 1);
  this->X_test = X_test;
  this->Y_test = Y_test;
  this->X_train = X_train;
  this->Y_train = Y_train;
  // Set up X_train_scaled and Y_train_scaled etc as original X_train and
  // Y_train etc. We reset it in test_train_split
  this->is_scaled = false;
  this->X_train_scaled = X_train;
  this->Y_train_scaled = Y_train;
  this->X_test_scaled = X_test;
  this->Y_test_scaled = Y_test;
}

double LogisticRegression::calculate_hypthesis(Eigen::VectorXd data) {
  double res = this->thetas(0);
  for (auto i = 0; i < data.size(); i++) {
    res += data(i) * this->thetas(i + 1);
  }
  // Pass our parameters through the sigmoid activation function
  return static_cast<double>(1.0 / (1.0 + std::exp(-res)));
}
} // namespace ml
