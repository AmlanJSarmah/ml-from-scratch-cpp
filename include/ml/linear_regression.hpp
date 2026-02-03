#pragma once

#include "ml/model.hpp"
#include <Eigen/Dense>

namespace ml {
class LinearRegression : public Model {
  Eigen::VectorXd thetas;
  double calculate_hypothesis(Eigen::VectorXd row);

public:
  LinearRegression(const Eigen::MatrixXd &X_train,
                   const Eigen::VectorXd &Y_train,
                   const Eigen::MatrixXd &X_test,
                   const Eigen::VectorXd &Y_test);
  void fit();
  void test();
  double predict(Eigen::VectorXd data);
};
} // namespace ml
