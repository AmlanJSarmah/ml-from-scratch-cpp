#pragma once
#include "Eigen/Dense"
#include "ml/model.hpp"

namespace ml {
class LogisticRegression : public Model {
  double learning_rate;
  double max_iter;
  Eigen::VectorXd thetas;
  double calculate_hypthesis(Eigen::VectorXd data);

public:
  LogisticRegression(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
                     Eigen::MatrixXd X_test, Eigen::VectorXd Y_test,
                     double learning_rate = 0.01, double max_iter = 1000);
};
} // namespace ml
