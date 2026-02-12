#include "ml/naive_bayes.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage linear_regression <dataset.csv path "
                 "<target_column_number> <is_target_column_a_string(enter 1/0)>"
              << std::endl;
    return 1;
  }

  // Load CSV
  try {
    auto data = (std::string(argv[3]).compare("1") == 0)
                    ? load_csv(argv[1], std::stoi(argv[2]), true)
                    : load_csv(argv[1], std::stoi(argv[2]), false);
    // View our data
    data.print_dataset(5);
    // Splitting
    const auto &[train, test] = test_train_split(0.2, data);

    const auto &X_train = train.first;
    const auto &Y_train = train.second;

    const auto &X_test = test.first;
    const auto &Y_test = test.second;

  } catch (std::string err) {
    std::cerr << err << std::endl;
    return 1;
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what()
              << " Probably error in 'is_target_column_a_string' or "
                 "'wrong_target_column_selected'"
              << "\n";
    return 1;
  }

  return 0;
}
