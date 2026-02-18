#include "ml/k_means.hpp"
#include <Eigen/Dense>

namespace ml {
KMeans::KMeans(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
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
} // namespace ml
