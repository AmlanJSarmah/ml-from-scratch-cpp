# ml-from-scratch-cpp
Implementation of various supervised and unsupervised machine learning algorithm in C++

## Setup instructions.
1. It is assumed that `cmake` and `g++` is installed.  
2. Run the following commands in the project directiory
```
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j
```
3. We have now build the project in the `build` directory you will see various executables like `linear_regression`.
4. To run Linear Regression from `build` folder
```
linear_regression ../data/iris_dataset.csv 5
```
Here, `../data/iris_dataset.csv` is path to dataset and `5` is the column where target variable is located.

## Features Implemented and Demonstration
### CSV Parser
The `load_csv()` function parses a CSV file and stores the numberical data while elegantly formatting it to be displayed on the console using `.print_dataset()` function.  

<img width="872" height="597" alt="image" src="https://github.com/user-attachments/assets/3218b0c9-f816-44e0-b9b3-94096ed8715c" />

### Splitting data
We can now use `test_train_splie()` to split testing and training data.  
<img width="876" height="243" alt="image" src="https://github.com/user-attachments/assets/92a9f1ee-15dc-48a4-8917-1758ca5bae6f" />
