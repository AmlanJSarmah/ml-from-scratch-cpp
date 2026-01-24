#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

std::vector<std::vector<double>> load_csv(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<std::vector<double>> data;
  std::string line;
  while (std::getline(file, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(clean_number_str(cell)));
    }
    data.push_back(row);
  }
  return data;
}
