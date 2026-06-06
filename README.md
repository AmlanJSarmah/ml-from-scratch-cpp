# ml-from-scratch-cpp
Live demo: https://ml-from-scratch-cpp.vercel.app/

Implementation of various supervised and unsupervised machine learning algorithms in C++.

## Setup instructions
1. Prerequisites: `cmake`, `g++` (or another C++ compiler) and standard build tools.
2. From the project root run:

```
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j
```

3. The build artifacts are placed in the `build` directory. You should see executables such as `linear_regression`.

4. Example: run Linear Regression from the `build` folder

```
./linear_regression ../data/LinearRegression/california_housing.csv 9 0
```

Where `../data/LinearRegression/california_housing.csv` is the dataset path, `9` is the target column index, and `0` indicates the target is numeric.

See the `data` directory for example datasets.