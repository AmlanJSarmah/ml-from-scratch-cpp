#pragma once

#include <Eigen/Dense>

namespace ml {
class LinearRegression {
  double calculate_hypothesis(Eigen::VectorXd thetas, Eigen::VectorXd row);

public:
  Eigen::VectorXd thetas;
  Eigen::MatrixXd training_data;
  Eigen::VectorXd testing_data;
  double learning_rate;
  LinearRegression(const Eigen::MatrixXd &training_data,
                   const Eigen::VectorXd &testing_data,
                   double learning_rate = 0.01);
};
} // namespace ml
