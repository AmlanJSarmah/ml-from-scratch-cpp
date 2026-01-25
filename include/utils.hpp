#pragma once

#include <map>
#include <string>
#include <sys/types.h>
#include <vector>

class Dataset {
  std::map<double, std::string> target_str_values;
  bool is_target_str;

public:
  std::vector<std::vector<double>> features;
  std::vector<double> target;
  Dataset(std::vector<std::vector<double>> features,
          std::vector<double> target);
  Dataset(std::vector<std::vector<double>> features, std::vector<double> target,
          std::map<double, std::string> target_str_values);
  void print_dataset();
};

Dataset load_csv(const std::string &filepath, size_t target_column_idx,
                 bool is_target_str);
