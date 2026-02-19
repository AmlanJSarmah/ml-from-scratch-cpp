#pragma once

#include "ml/model.hpp"

namespace ml {
class KMeans : public Model {
  Eigen::MatrixXd centroids;
  Eigen::VectorXi labels;
  int k;
  int max_iters;

public:
  KMeans(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
         Eigen::MatrixXd X_test, Eigen::VectorXd Y_test, int k,
         int max_iters = 1000);

  void fit();
  void test();
};
} // namespace ml
