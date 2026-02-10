#include "ml/logistic_regression.hpp"
#include "Eigen/Dense"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace ml {
// NOTE: We have stored the value of Y_train_scaled and Y_test_scaled.
// These values are not to be used with logistic regression.
// test_train_scaled() is written in such a way that it scaled X_test, X_train,
// Y_test and Y_train
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

  for (int iter = 0; iter < this->max_iter; iter++) {
    // Compute predictions
    Eigen::VectorXd predictions = calculate_all_hypotheses();

    // Compute gradients
    Eigen::VectorXd errors = predictions - this->Y_train_scaled;
    Eigen::VectorXd gradients = (X_with_bias.transpose() * errors) / m;

    // Update parameters
    this->thetas -= this->learning_rate * gradients;

    // Compute cost for convergence check
    double current_cost = compute_cost();

    // Check for convergence
    if (std::abs(prev_cost - current_cost) < tolerance) {
      std::cout << "Converged at iteration " << iter << std::endl;
      std::cout << "Final cost: " << current_cost << std::endl;
      break;
    }

    prev_cost = current_cost;
  }
}

void LogisticRegression::test() {
  std::cout << "========== TEST ==========" << std::endl;

  int true_positive = 0, true_negative = 0;
  int false_positive = 0, false_negative = 0;
  int correct = 0, incorrect = 0;

  double total_loss = 0.0;
  // Small epsilon to prevent log(0)
  double eps = 1e-15;

  for (int i = 0; i < X_test_scaled.rows(); i++) {
    Eigen::VectorXd v = this->X_test_scaled.row(i).transpose();

    // Get probability prediction (0 to 1)
    double predicted_prob = this->calculate_hypothesis(v);

    // Get binary prediction (0 or 1)
    int predicted_class = (predicted_prob >= 0.5) ? 1 : 0;

    // Get actual class
    int actual_class = static_cast<int>(this->Y_test(i));

    // Update confusion matrix
    if (actual_class == 1 && predicted_class == 1)
      true_positive++;
    else if (actual_class == 0 && predicted_class == 0)
      true_negative++;
    else if (actual_class == 0 && predicted_class == 1)
      false_positive++;
    else if (actual_class == 1 && predicted_class == 0)
      false_negative++;

    // Count correct/incorrect
    if (predicted_class == actual_class)
      correct++;
    else
      incorrect++;

    // Calculate log loss (binary cross-entropy)
    // Clip predictions to prevent log(0)
    double clipped_prob = std::max(eps, std::min(1.0 - eps, predicted_prob));
    total_loss += -(actual_class * std::log(clipped_prob) +
                    (1 - actual_class) * std::log(1 - clipped_prob));
  }

  int n = X_test_scaled.rows();

  // Calculate metrics
  double accuracy = (static_cast<double>(correct) / n) * 100.0;
  double log_loss = total_loss / n;

  // Precision, Recall, F1-Score
  double precision = (true_positive + false_positive > 0)
                         ? static_cast<double>(true_positive) /
                               (true_positive + false_positive)
                         : 0.0;

  double recall = (true_positive + false_negative > 0)
                      ? static_cast<double>(true_positive) /
                            (true_positive + false_negative)
                      : 0.0;

  double f1_score = (precision + recall > 0)
                        ? 2 * (precision * recall) / (precision + recall)
                        : 0.0;

  // Print results
  std::cout << "Number of accurate predictions: " << correct << " out of " << n
            << std::endl;
  std::cout << "Accuracy    : " << accuracy << " %" << std::endl;
  std::cout << "Log Loss    : " << log_loss << std::endl;
  std::cout << std::endl;

  std::cout << "Confusion Matrix:" << std::endl;
  std::cout << std::setw(15) << "Predicted Negative | Predicted Positive"
            << std::endl;
  std::cout << "Actual Negative" << std::setw(10) << true_negative
            << std::setw(10) << "|" << std::setw(10) << false_positive
            << std::endl;
  std::cout << "Actual Positive" << std::setw(10) << false_negative
            << std::setw(10) << "|" << std::setw(10) << true_positive
            << std::endl;
  std::cout << std::endl;

  std::cout << "Precision   : " << precision << " (" << (precision * 100)
            << "%)" << std::endl;
  std::cout << "Recall      : " << recall << " (" << (recall * 100) << "%)"
            << std::endl;
  std::cout << "F1-Score    : " << f1_score << std::endl;
}

double LogisticRegression::predict(Eigen::VectorXd data) {
  return calculate_hypothesis(data);
}
} // namespace ml
