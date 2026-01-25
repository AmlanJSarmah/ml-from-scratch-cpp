#include "ml/linear_regression.hpp"
#include "utils.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage linear_regression <dataset.csv> <target_column_number>"
              << std::endl;
    return 1;
  }

  // Load CSV
  try {
    auto data = load_csv(argv[1], std::stoi(argv[2]), true);
    // View our data
    data.print_dataset();
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
