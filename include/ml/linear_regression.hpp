#pragma once

#include <Eigen/Dense>

namespace ml {
class LinearRegression {
  double calculate_hypothesis(Eigen::VectorXd row);

public:
  Eigen::VectorXd thetas;
  Eigen::MatrixXd X_train;
  Eigen::VectorXd Y_train;
  Eigen::MatrixXd X_test;
  Eigen::VectorXd Y_test;
  Eigen::MatrixXd X_train_scaled;
  Eigen::VectorXd Y_train_scaled;
  Eigen::MatrixXd X_test_scaled;
  Eigen::VectorXd Y_test_scaled;
  Eigen::VectorXd X_train_means;
  Eigen::VectorXd X_train_stds;
  double Y_train_means;
  double Y_train_stds;
  double learning_rate;
  double number_of_epochs;
  LinearRegression(const Eigen::MatrixXd &X_train,
                   const Eigen::VectorXd &Y_train,
                   const Eigen::MatrixXd &X_test, const Eigen::VectorXd &Y_test,
                   double learning_rate = 0.01, double number_of_epochs = 1000);
  void train_ne();
  void test();
  double predict(Eigen::VectorXd data);
};
} // namespace ml
