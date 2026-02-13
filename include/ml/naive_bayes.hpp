#pragma once

#include "Eigen/Dense"
#include "ml/model.hpp"

namespace ml {
class NaiveBayes : public Model {
  // For laplace smoothing
  double var_laplace_smoothing = 1e-9;

  // ======== For Gaussian Naive Bayes ========
  // Store means of per feature per class
  // shape: (num_classes, n_features)
  Eigen::MatrixXd means;
  // Store variance of per feature per class
  // shape: (num_classes, n_features)
  Eigen::MatrixXd variances;
  // Stores the prior
  // shape: (num_classes)
  Eigen::VectorXd priors;

public:
  NaiveBayes(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
             Eigen::MatrixXd X_test, Eigen::VectorXd Y_test);
  void fit();
  int predict(const Eigen::VectorXd &data);
  void test();
};
} // namespace ml
