#include "utils.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "ml/model.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ========== Dataset Class methods ==========
Dataset::Dataset(std::vector<std::vector<double>> features,
                 std::vector<double> target,
                 std::vector<std::string> feature_names) {
  this->features = features;
  this->target = target;
  this->is_target_str = false;
  this->feature_names = feature_names;

  // Create a Matrix for efficient calculation later
  size_t rows = this->features.size();
  size_t cols = this->features.at(0).size();
  Eigen::MatrixXd f(rows, cols);
  Eigen::VectorXd v(rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      f(i, j) = this->features[i][j];
    }
    v(i) = this->target[i];
  }

  this->_features = f;
  this->_target = v;
}

Dataset::Dataset(std::vector<std::vector<double>> features,
                 std::vector<double> target,
                 std::vector<std::string> feature_names,
                 std::map<double, std::string> target_str_values) {
  this->features = features;
  this->target = target;
  this->target_str_values = target_str_values;
  this->is_target_str = true;
  this->feature_names = feature_names;

  // Create a Matrix for efficient calculation later
  size_t rows = this->features.size();
  size_t cols = this->features.at(0).size();
  Eigen::MatrixXd f(rows, cols);
  Eigen::VectorXd v(rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      f(i, j) = this->features[i][j];
    }
    v(i) = this->target[i];
  }

  this->_features = f;
  this->_target = v;
}

void Dataset::print_dataset(int n_rows) const {
  size_t cols_to_print =
      this->features[0].size() > 8 ? 8 : this->features[0].size();
  n_rows = n_rows == -1 ? this->features.size() : n_rows;

  std::cout << "========== DATASET " << n_rows << "-rows "
            << "===========" << std::endl;
  std::cout << std::fixed << std::setprecision(3);

  // Print header
  for (size_t i = 0; i < cols_to_print; i++) {
    std::cout << this->feature_names.at(i) << std::setw(20);
  }
  std::cout << "\n";

  // Print actual data
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < cols_to_print; j++) {
      std::cout << this->features[i][j] << std::setw(20);
    }
    if (is_target_str)
      std::cout << this->target_str_values.find(this->target.at(i))->second
                << std::setw(20);
    else
      std::cout << target.at(i) << std::setw(20);
    std::cout << "\n";
  }

  std::cout.unsetf(std::ios::fixed);
  std::cout << std::setprecision(6);
  if (cols_to_print < this->features[0].size()) {
    std::cout << "Showing first 8 columns and target" << std::endl;
  }

  std::cout << "=========== END ==========" << std::endl;
}

// ========== CSV Utilities ==========
std::string clean_number_str(std::string s) {
  // Remove surrounding double quotes if present
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    s = s.substr(1, s.size() - 2);
  }

  // Trim leading/trailing whitespace (including tabs, newlines, etc.)
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "0.0"; // or throw, or return ""
  }
  size_t end = s.find_last_not_of(" \t\r\n");
  s = s.substr(start, end - start + 1);
  return s;
}

Dataset load_csv(const std::string &filename, size_t target_column_idx,
                 bool is_target_str) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<std::vector<double>> features;
  std::vector<std::string> _target;
  std::vector<double> target;
  std::vector<std::string> feature_names;
  std::string line;
  size_t line_counter = 0;

  // Extract features and target via traversing the file
  while (std::getline(file, line)) {
    line_counter++;

    // Ignore the header column
    if (line_counter == 1) {
      std::stringstream titles(line);
      std::string title;
      while (std::getline(titles, title, ','))
        feature_names.push_back(title);
      continue;
    }

    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    size_t column_idx = 0;

    // Iterating over the columns
    while (std::getline(ss, cell, ',')) {
      column_idx++;
      // If target column then push to target vector
      if (column_idx == target_column_idx) {
        _target.push_back(clean_number_str(cell));
        continue;
      }
      row.push_back(std::stod(clean_number_str(cell)));
    }
    features.push_back(row);
  }
  file.close();

  // If target is already numerical in dataset push to target directly
  if (!is_target_str) {
    for (size_t i = 0; i < _target.size(); i++) {
      target.push_back(std::stod(_target.at(i)));
    }
    return Dataset(features, target, feature_names);
  }

  // If target is string encode it to numerical
  else {
    std::set<std::string> unique_target(_target.begin(), _target.end());
    std::map<double, std::string> target_str_values;
    int counter = 0;
    for (auto it = unique_target.cbegin(); it != unique_target.cend(); it++) {
      target_str_values.insert(std::pair<double, std::string>{counter, *it});
      counter++;
    }
    for (size_t i = 0; i < _target.size(); i++) {
      for (auto element : target_str_values) {
        if (element.second == _target.at(i)) {
          target.push_back(element.first);
        }
      }
    }
    return Dataset(features, target, feature_names, target_str_values);
  }
}

// ============ DATA SPLITTING ============

std::pair<std::pair<Eigen::MatrixXd, Eigen::VectorXd>,
          std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
test_train_split(float ratio, const Dataset &d) {
  if (d.features.empty())
    throw std::string("Empty Dataset");

  // Setup Eigen
  int cols = d._features.cols();
  int rows = d._features.rows();
  int rows_test = static_cast<int>(rows * ratio);
  int rows_train = rows - rows_test;
  Eigen::MatrixXd train_features(rows_train, cols);
  Eigen::MatrixXd test_features(rows_test, cols);
  Eigen::VectorXd train_target(rows_train);
  Eigen::VectorXd test_target(rows_test);

  // Allocation
  for (int i = 0; i < rows_train; ++i) {
    for (int j = 0; j < cols; ++j) {
      train_features(i, j) = d._features(i, j);
    }
    train_target(i) = d._target(i);
  }
  for (int i = 0; i < rows_test; ++i) {
    for (int j = 0; j < cols; ++j) {
      test_features(i, j) = d._features(rows_train + i, j);
    }
    test_target(i) = d._target(rows_train + i);
  }

  return {{train_features, train_target}, {test_features, test_target}};
}

// ========== SCALING ============
void standard_scalar(ml::Model &lr) {
  // Test
  Eigen::RowVectorXd mean_X_train = lr.X_train.colwise().mean();
  Eigen::RowVectorXd std =
      ((lr.X_train.rowwise() - mean_X_train).array().square().colwise().mean())
          .sqrt();
  lr.X_train_scaled =
      (lr.X_train.rowwise() - mean_X_train).array().rowwise() / std.array();
  lr.Y_train_scaled =
      (lr.Y_train.array() - lr.Y_train.mean()) /
      std::sqrt((lr.Y_train.array() - lr.Y_train.mean()).square().mean());
  lr.X_train_means = mean_X_train;
  lr.X_train_stds = std;
  lr.Y_train_means = lr.Y_train.mean();
  lr.Y_train_stds =
      std::sqrt((lr.Y_train.array() - lr.Y_train_means).square().sum() /
                (lr.Y_train.size() - 1));
  // Training
  Eigen::RowVectorXd mean_X_test = lr.X_test.colwise().mean();
  Eigen::RowVectorXd _std =
      ((lr.X_test.rowwise() - mean_X_test).array().square().colwise().mean())
          .sqrt();
  lr.X_test_scaled =
      (lr.X_test.rowwise() - mean_X_test).array().rowwise() / _std.array();
  lr.Y_test_scaled =
      (lr.Y_test.array() - lr.Y_test.mean()) /
      std::sqrt((lr.Y_test.array() - lr.Y_test.mean()).square().mean());
  // After scaling X_train, X_test, Y_train and Y_test we set is_scaled as true
  lr.is_scaled = true;
}
