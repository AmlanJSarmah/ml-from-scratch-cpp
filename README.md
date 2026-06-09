# ml-from-scratch-cpp
Live demo: https://ml-from-scratch-cpp.vercel.app/  

Implementation of various supervised and unsupervised machine learning algorithms in C++.  

**Note** : All the datasets won't work with all the models, for example running `naive_bayes`, `k_means` or `logistic_regression` on `california_housing` will throw error as the `target` feature(column) in the dataset are *continuous* values not ideal for classification & clustering tasks but for regression tasks.  

## Demo
https://github.com/user-attachments/assets/a17955ac-3700-4e0a-b577-30d534833e09


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
./linear_regression ../data/california_housing.csv 9 0
```

Where `../data/california_housing.csv` is the dataset path, `9` is the target column index, and `0` indicates the target is numeric.

See the `data` directory for example datasets.


## Using the models via the command line  
1. For classification & regression models, the fromat for command line args are  
```
./model_name ../data/dataset.csv target_column_number is_target_index_number(0/1)  
```
2. For clustering models, the format is  
```
./model_name ../data/dataset.csv target_column_number is_target_index_numer(0/1) number_of_cluster  
```
**Example**   
The dataset `iris` the target feature(column) is the 5th column here the `Species` also the target column is a string, with values `Iris-setosa`, `Iris-versicolor` and `Iris-virginica`.  
So the command will be `./model ../data/iris.csv 5 1` for regression & classification and `./model ../data/iris.csv 5 3 1` for clustering.  
