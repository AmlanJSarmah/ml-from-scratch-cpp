#include "ml/naive_bayes.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if ((argc != 4) && (argc != 5)) {
    std::cerr << "Usage naive_bayes <dataset.csv path "
                 "<target_column_number> <is_target_column_a_string(enter "
                 "1/0)> <benchmark_mode(enter 1/0)(optional)>"
              << std::endl;
    return 1;
  }

  bool benchmark_mode = 0;
  if (argc == 5) {
    benchmark_mode = (std::string(argv[4]).compare("1") == 0) ? 1 : 0;
  }

  // Load CSV
  try {
    auto data = (std::string(argv[3]).compare("1") == 0)
                    ? load_csv(argv[1], std::stoi(argv[2]), true)
                    : load_csv(argv[1], std::stoi(argv[2]), false);
    // View our data
    if (!benchmark_mode)
      data.print_dataset(5);
    // Splitting
    const auto &[train, test] = test_train_split(0.2, data);

    const auto &X_train = train.first;
    const auto &Y_train = train.second;

    const auto &X_test = test.first;
    const auto &Y_test = test.second;

    // Model training and testing
    ml::NaiveBayes NaiveBayes(X_train, Y_train, X_test, Y_test);
    NaiveBayes.fit();
    if (benchmark_mode) {
      NaiveBayes.test_benchmark();
      return 0;
    }
    NaiveBayes.test();
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
