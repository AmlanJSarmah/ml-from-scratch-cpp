#include "ml/k_means.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>

namespace ml {
KMeans::KMeans(Eigen::MatrixXd X_train, Eigen::VectorXd Y_train,
               Eigen::MatrixXd X_test, Eigen::VectorXd Y_test, int k,
               int max_iters) {
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
  // Init Centroids
  this->centroids = Eigen::MatrixXd::Zero(k, X_train.cols());
  this->labels = Eigen::VectorXi::Zero(X_train.rows());
  // Other parameters
  this->k = k;
  this->max_iters = max_iters;
}

void KMeans::fit() {
  int n = this->X_train_scaled.rows();
  int features = this->X_train_scaled.cols();

  // Randomly initialize centroids by picking k random training points
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(),
               std::mt19937{std::random_device{}()});

  centroids = Eigen::MatrixXd(k, features);
  for (int i = 0; i < k; i++) {
    centroids.row(i) = this->X_train_scaled.row(indices[i]);
  }

  Eigen::VectorXi labels(n);

  for (int iter = 0; iter < max_iters; iter++) {
    // Assignment step: assign each point to nearest centroid
    for (int i = 0; i < n; i++) {
      double best_dist = std::numeric_limits<double>::infinity();
      int best_cluster = 0;
      for (int j = 0; j < k; j++) {
        double dist =
            (this->X_train_scaled.row(i) - centroids.row(j)).squaredNorm();
        if (dist < best_dist) {
          best_dist = dist;
          best_cluster = j;
        }
      }
      labels(i) = best_cluster;
    }

    // Update step: recompute centroids as mean of assigned points
    Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, features);
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);

    for (int i = 0; i < n; i++) {
      new_centroids.row(labels(i)) += this->X_train_scaled.row(i);
      counts(labels(i))++;
    }

    for (int j = 0; j < k; j++) {
      if (counts(j) > 0)
        new_centroids.row(j) /= counts(j);
    }

    // Check convergence
    double tol = 1e-4;
    if ((new_centroids - centroids).norm() < tol)
      break;

    centroids = new_centroids;
  }

  // Store final labels
  this->labels = labels;
}
void KMeans::test() {
  int n = this->X_test_scaled.rows();
  int correct = 0;

  for (int i = 0; i < n; i++) {
    // Find nearest centroid
    double best_dist = std::numeric_limits<double>::infinity();
    int best_cluster = 0;
    for (int j = 0; j < k; j++) {
      double dist =
          (this->X_test_scaled.row(i) - centroids.row(j)).squaredNorm();
      if (dist < best_dist) {
        best_dist = dist;
        best_cluster = j;
      }
    }

    if (best_cluster == (int)this->Y_test_scaled(i))
      correct++;
    std::cout << best_cluster << " : " << this->Y_test_scaled(i) << std::endl;
  }

  double accuracy = (double)correct / n * 100.0;
  std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
}
} // namespace ml
