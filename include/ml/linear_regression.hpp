#pragma once

#include "ml/model.hpp"
#include <Eigen/Dense>

namespace ml {
class LinearRegression : public Model {
  double calculate_hypothesis(Eigen::VectorXd row);

public:
  Eigen::VectorXd thetas;
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
