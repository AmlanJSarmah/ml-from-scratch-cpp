#include "ml/linear_regression.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace ml {
LinearRegression::LinearRegression(const Eigen::MatrixXd &X_train,
                                   const Eigen::VectorXd &Y_train,
                                   const Eigen::MatrixXd &X_test,
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
                 X_with_bias.transpose() * this->Y_train;
}

void LinearRegression::test() {
  std::cout << "========== TEST ==========" << std::endl;
  float accuracy;
  float correct = 0, incorrect = 0;
  double mse = 0, mae = 0, ss_res = 0, ss_tot = 0;

  double y_mean = Y_test.mean();

  for (auto i = 0; i < X_test.rows(); i++) {
    Eigen::VectorXd v = this->X_test.row(i).transpose();
    auto predicted_ = calculate_hypothesis(this->thetas, v);
    auto predicted = (predicted_);
    auto actual = (Y_test(i));

    double error = actual - predicted;

    double tolerance = 0.5; // Within 0.5 standard deviations
    if (std::abs(error) < tolerance)
      correct++;
    else
      incorrect++;

    mse += error * error;
    mae += std::abs(error);
    ss_res += error * error;
    ss_tot += (actual - y_mean) * (actual - y_mean);
  }

  int n = X_test.rows();
  accuracy = (correct / static_cast<float>(X_test.rows())) * 100;
  mse /= n;
  mae /= n;
  double rmse = std::sqrt(mse);
  double r2 = 1.0 - (ss_res / ss_tot);

  std::cout << "Number of accurate prediction " << correct << " out of "
            << X_test.rows() << std::endl;
  std::cout << "Accuracy is : " << accuracy << " %" << std::endl;
  std::cout << "R² Score    : " << r2 << std::endl;
  std::cout << "RMSE        : " << rmse << std::endl;
  std::cout << "MAE         : " << mae << std::endl;
  std::cout << "MSE         : " << mse << std::endl;
}
} // namespace ml
