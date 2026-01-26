#include "ml/linear_regression.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace ml {
LinearRegression::LinearRegression(const Eigen::MatrixXd &X_train,
                                   const Eigen::VectorXd &X_test,
                                   const Eigen::MatrixXd &Y_train,
                                   const Eigen::VectorXd &Y_test,
                                   double learning_rate,
                                   double number_of_epochs) {
  this->X_test = X_test;
  this->X_train = X_train;
  this->Y_test = Y_test;
  this->Y_train = Y_train;
  this->thetas = Eigen::VectorXd::Zero(static_cast<int>(X_train.cols()) + 1);
  this->learning_rate = learning_rate;
  this->number_of_epochs = number_of_epochs;
}

double LinearRegression::calculate_hypothesis(Eigen::VectorXd thetas,
                                              Eigen::VectorXd row) {
  double res = thetas(0);
  for (auto i = 0; i < row.size(); i++) {
    res += row(i) * thetas(i + 1);
  }
  return res;
}

void LinearRegression::train_ne() {
  // we use normal equation
  // Account for Bias
  Eigen::MatrixXd X_with_bias(this->X_train.rows(), this->X_train.cols() + 1);
  X_with_bias.col(0) = Eigen::VectorXd::Ones(this->X_train.rows());
  X_with_bias.rightCols(this->X_train.cols()) = this->X_train;
  // Normal Equation
  this->thetas = (X_with_bias.transpose() * X_with_bias).inverse() *
                 X_with_bias.transpose() * this->X_test;
}

void LinearRegression::test() {
  std::cout << "========== TEST ==========" << std::endl;
  float accuracy;
  float correct = 0, incorrect = 0;
  for (auto i = 0; i < Y_train.rows(); i++) {
    Eigen::VectorXd v = this->Y_train.row(i).transpose();
    auto predicted_ = calculate_hypothesis(this->thetas, v);
    auto predicted = static_cast<int>(predicted_);
    auto actual = static_cast<int>(Y_test(i));
    if (actual == predicted)
      correct++;
    else
      incorrect++;
    std::cout << predicted_ << " : " << predicted << " : " << actual
              << std::endl;
  }
  accuracy = (correct / static_cast<float>(Y_train.rows())) * 100;
  std::cout << "Number of accurate prediction " << correct << " out of "
            << Y_train.rows() << std::endl;
  std::cout << "Accuracy is : " << accuracy << " % " << std::endl;
}
} // namespace ml
