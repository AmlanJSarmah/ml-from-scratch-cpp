#pragma once
#include "Eigen/Dense"
#include "ml/model.hpp"

namespace ml {
class LogisticRegression : public Model {
  double learning_rate;
  double max_iter;
  Eigen::VectorXd thetas;
  double calculate_hypothesis(Eigen::VectorXd data);
  Eigen::VectorXd calculate_all_hypotheses();
  double compute_cost();

public:
  LogisticRegression(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
                     Eigen::MatrixXd X_test, Eigen::VectorXd Y_test,
                     double learning_rate = 0.01, double max_iter = 10000);
  void fit();
  void test();
  double predict(Eigen::VectorXd data);
};
} // namespace ml
