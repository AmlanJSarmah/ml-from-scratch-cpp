#include "ml/naive_bayes.hpp"
#include "Eigen/Dense"
#include <iostream>

namespace ml {
NaiveBayes::NaiveBayes(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
                       Eigen::MatrixXd X_test, Eigen::VectorXd Y_test) {

  this->X_train = X_train;
  this->X_test = X_test;
  this->Y_train = Y_train;
  this->Y_test = Y_test;
  // Set up X_train_scaled and Y_train_scaled etc as original X_train and
  // Y_train etc. We reset it in test_train_split
  // We use these scaled variables for all computations, and in case they aren't
  // actually scaled, the values are populated
  this->is_scaled = false;
  this->X_train_scaled = X_train;
  this->Y_train_scaled = Y_train;
  this->X_test_scaled = X_test;
  this->Y_test_scaled = Y_test;
}

void NaiveBayes::fit() {
  // m is number of data points
  // n is number of features
  int m = X_train_scaled.rows();
  int n = X_train_scaled.cols();

  // Assume binary classification and set initial values of priors, mean and
  // variance per features per class
  int n_classes = 2;
  this->means = Eigen::MatrixXd::Zero(n_classes, n);
  this->variances = Eigen::MatrixXd::Zero(n_classes, n);
  this->priors = Eigen::VectorXd::Zero(n_classes);

  // Assume we have binary classification with 10 +ve and 5 -ve so it stores {5,
  // 10}
  // Helps achive a lot of funtion of the indicator function
  Eigen::VectorXd class_counts = Eigen::VectorXd::Zero(n_classes);

  for (int i = 0; i < m; ++i) {
    int c = static_cast<int>(this->Y_train(i));
    class_counts(c) += 1.0;
    // used later for mean
    this->means.row(c) += this->X_train_scaled.row(i);
  }

  // We compute the means
  for (int c = 0; c < n_classes; ++c) {
    if (class_counts(c) > 0) {
      this->means.row(c) /= class_counts(c);
    }
  }

  // We compute variance
  for (int i = 0; i < m; ++i) {
    int c = static_cast<int>(this->Y_train(i));
    Eigen::RowVectorXd diff = this->X_train_scaled.row(i) - this->means.row(c);
    this->variances.row(c) += diff.array().square().matrix();
  }

  for (int c = 0; c < n_classes; ++c) {
    if (class_counts(c) > 0) {
      this->variances.row(c) /= class_counts(c);
    }
  }

  // We perform laplace smoothning
  this->variances.array() += var_laplace_smoothing;

  // We compute prios
  this->priors = class_counts / m;
}

}; // namespace ml
