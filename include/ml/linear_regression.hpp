#pragma once

#include <Eigen/Dense>

namespace ml {
class LinearRegression {
  double calculate_hypothesis(Eigen::VectorXd thetas, Eigen::VectorXd row);

public:
  Eigen::VectorXd thetas;
  Eigen::MatrixXd X_train;
  Eigen::VectorXd X_test;
  Eigen::MatrixXd Y_train;
  Eigen::VectorXd Y_test;
  double learning_rate;
  double number_of_epochs;
  LinearRegression(const Eigen::MatrixXd &X_train,
                   const Eigen::VectorXd &X_test,
                   const Eigen::MatrixXd &Y_train,
                   const Eigen::VectorXd &Y_test, double learning_rate = 0.01,
                   double number_of_epochs = 1000);
  void train_ne();
  void test();
};
} // namespace ml
