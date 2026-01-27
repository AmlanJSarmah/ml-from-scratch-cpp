#pragma once

#include <Eigen/Dense>
#include <map>
#include <string>
#include <sys/types.h>
#include <vector>

class Dataset {
  std::map<double, std::string> target_str_values;
  bool is_target_str;

public:
  Eigen::MatrixXd _features;
  Eigen::VectorXd _target;
  Eigen::MatrixXd scaled_features;
  Eigen::VectorXd scaled_target;
  std::vector<std::vector<double>> features;
  std::vector<double> target;
  Dataset(std::vector<std::vector<double>> features,
          std::vector<double> target);
  Dataset(std::vector<std::vector<double>> features, std::vector<double> target,
          std::map<double, std::string> target_str_values);
  void print_dataset(int n_rows = -1) const;
  void standard_scalar();
};

Dataset load_csv(const std::string &filepath, size_t target_column_idx,
                 bool is_target_str);

std::pair<std::pair<Eigen::MatrixXd, Eigen::VectorXd>,
          std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
test_train_split(float ratio, const Dataset &d);
