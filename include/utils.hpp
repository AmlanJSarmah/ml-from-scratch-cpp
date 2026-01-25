#pragma once

#include <string>
#include <sys/types.h>
#include <vector>

std::vector<std::vector<double>> load_csv(const std::string &filepath,
                                          size_t feature_column);
