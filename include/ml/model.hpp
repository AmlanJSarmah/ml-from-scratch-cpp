#pragma once

#include <Eigen/Dense>

namespace ml {
class Model {
public:
  bool is_scaled;
  // Data Metrics
  Eigen::MatrixXd X_train;
  Eigen::VectorXd Y_train;
  Eigen::MatrixXd X_test;
  Eigen::VectorXd Y_test;
  // Scaled data metrics
  Eigen::MatrixXd X_train_scaled;
  Eigen::VectorXd Y_train_scaled;
  Eigen::MatrixXd X_test_scaled;
  Eigen::VectorXd Y_test_scaled;
  Eigen::VectorXd X_train_means;
  Eigen::VectorXd X_train_stds;
  double Y_train_means;
  double Y_train_stds;
};
} // namespace ml
