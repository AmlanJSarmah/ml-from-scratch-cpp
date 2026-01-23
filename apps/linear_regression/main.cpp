#include <iostream>
#include "ml/linear_regression.hpp"

int main() {
  ml::LinearRegression model;
  std::cout << model.calculate_hypothesis() << std::endl;
  return 0;
}
