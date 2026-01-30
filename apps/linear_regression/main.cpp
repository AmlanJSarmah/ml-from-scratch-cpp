#include "ml/linear_regression.hpp"
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

    // Linear Regression
    ml::LinearRegression LinearRegression(X_train, Y_train, X_test, Y_test);
    // Scaling
    standard_scalar(LinearRegression);
    // Training and Testing
    LinearRegression.train_ne();
    LinearRegression.test();
    // Predicting in california housing
    std::string ds = argv[1];
    if (ds.find("california_housing_shuffled.csv") != std::string::npos) {
      Eigen::VectorXd x(8);
      // Suburban california
      x << 5.0, 20.0, 5.5, 1.1, 1200, 3.0, 34.2, -118.4;
      // High income bay Area
      // x << 9.5, 30.0, 6.5, 1.0, 900, 2.5, 37.8, -122.4;
      // Rural area
      // x << 2.8, 25.0, 4.8, 1.2, 3500, 3.5, 36.5, -119.5;
      std::cout << "Sample prediction for : ";
      for (auto i = 0; i < x.size(); i++) {
        std::cout << x(i) << " ";
      }
      std::cout << " is ";
      std::cout << LinearRegression.predict(x) << std::endl;
    }
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
