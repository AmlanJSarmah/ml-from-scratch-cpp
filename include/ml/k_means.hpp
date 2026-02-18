#pragma once

#include "ml/model.hpp"

namespace ml {
class KMeans : public Model {
  Eigen::VectorXd centroids;

public:
  KMeans(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
         Eigen::MatrixXd X_test, Eigen::VectorXd Y_test);
};
} // namespace ml
