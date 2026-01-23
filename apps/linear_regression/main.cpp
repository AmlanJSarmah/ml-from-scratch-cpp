#include <iostream>
#include "utils.hpp"
#include "ml/linear_regression.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage linear_regression <dataset.csv>" << std::endl;
    return 1;
  }

  // Load CSV
  try {
    auto data = load_csv(argv[1]);
    // View our data
    for (ssize_t i = 0; i < data.size(); i++) {
      for (ssize_t j = 0; j < data[i].size(); j++) {
	std::cout << data[i][j] << " ";
      }
      std::cout << "\n";
    }
  } catch (...) {
    std::cerr << "Error: Reading file" << std::endl;
    return 1;
  }

  ml::LinearRegression model;
  std::cout << model.calculate_hypothesis() << std::endl;
  return 0;
}
