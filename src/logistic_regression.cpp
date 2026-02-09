#include "ml/logistic_regression.hpp"
#include "Eigen/Dense"
#include <cmath>
#include <iostream>

namespace ml {
LogisticRegression::LogisticRegression(Eigen::MatrixXd X_train,
                                       Eigen::VectorXd Y_train,
                                       Eigen::MatrixXd X_test,
                                       Eigen::VectorXd Y_test,
                                       double learning_rate, double max_iter) {
  this->learning_rate = learning_rate;
  this->max_iter = max_iter;
  this->thetas = Eigen::VectorXd::Zero(static_cast<int>(X_train.cols()) + 1);
  this->X_test = X_test;
  this->Y_test = Y_test;
  this->X_train = X_train;
  this->Y_train = Y_train;
  // Set up X_train_scaled and Y_train_scaled etc as original X_train and
  // Y_train etc. We reset it in test_train_split
  this->is_scaled = false;
  this->X_train_scaled = X_train;
  this->Y_train_scaled = Y_train;
  this->X_test_scaled = X_test;
  this->Y_test_scaled = Y_test;
}

double LogisticRegression::calculate_hypothesis(Eigen::VectorXd data) {
  double res = this->thetas(0);
  for (auto i = 0; i < data.size(); i++) {
    res += data(i) * this->thetas(i + 1);
  }
  // Pass our parameters through the sigmoid activation function
  return static_cast<double>(1.0 / (1.0 + std::exp(-res)));
}

Eigen::VectorXd LogisticRegression::calculate_all_hypotheses() {
  int m = this->X_train_scaled.rows();
  Eigen::VectorXd hypotheses(m);
  for (int i = 0; i < m; i++) {
    hypotheses(i) = this->calculate_hypothesis(X_train_scaled.row(i));
  }
  return hypotheses;
}

double LogisticRegression::compute_cost() {
  int m = X_train_scaled.rows();
  Eigen::VectorXd hypotheses = this->calculate_all_hypotheses();

  double cost = 0.0;
  // Small epsilon to prevent log(0)
  double eps = 1e-15;

  for (int i = 0; i < m; i++) {
    // Clip hypothesis to avoid log(0) or log(1)
    double h = std::max(eps, std::min(1.0 - eps, hypotheses(i)));
    // Binary cross-entropy formula
    cost += -this->Y_train(i) * std::log(h) -
            (1 - this->Y_train(i)) * std::log(1 - h);
  }

  return cost / m;
}

void LogisticRegression::fit() {
  // Account for Bias
  Eigen::MatrixXd X_with_bias(this->X_train_scaled.rows(),
                              this->X_train_scaled.cols() + 1);
  X_with_bias.col(0) = Eigen::VectorXd::Ones(this->X_train_scaled.rows());
  X_with_bias.rightCols(this->X_train_scaled.cols()) = this->X_train_scaled;

  // Constants
  int m = this->X_train.rows();
  int n = this->X_train.cols();
  // Hyperparameters
  double tolerance = 1e-6;

  // Training loop with convergence check
  double prev_cost = std::numeric_limits<double>::max();

  // for (int iter = 0; iter < this->max_iter; iter++) {
  while (true) {
    // Compute predictions
    Eigen::VectorXd predictions = calculate_all_hypotheses();

    // Compute gradients
    Eigen::VectorXd errors = predictions - this->Y_train;
    Eigen::VectorXd gradients = (X_with_bias.transpose() * errors) / m;

    // Update parameters
    this->thetas -= this->learning_rate * gradients;

    // Compute cost for convergence check
    double current_cost = compute_cost();

    // Check for convergence
    if (std::abs(prev_cost - current_cost) < tolerance) {
      // std::cout << "Converged at iteration " << iter << std::endl;
      std::cout << "Final cost: " << current_cost << std::endl;
      break;
    }

    prev_cost = current_cost;
  }
}
} // namespace ml
