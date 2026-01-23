#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "utils.hpp"

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
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    return data;
}
