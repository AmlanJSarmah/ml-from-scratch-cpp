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
linear_regression ../data/california_housing_suffled.csv 9 0
```
Here, `../data/california_housing_suffled.csv` is path to dataset and `9` is the column where target variable is located and `0` clarifies that the target columnn in numerical.

## Features Implemented and Demonstration
### CSV Parser
The `load_csv()` function parses a CSV file and stores the numberical data while elegantly formatting it to be displayed on the console using `.print_dataset()` function.  

<img width="872" height="597" alt="image" src="https://github.com/user-attachments/assets/3218b0c9-f816-44e0-b9b3-94096ed8715c" />

### Splitting data
We can now use `test_train_splie()` to split testing and training data.  
<img width="876" height="243" alt="image" src="https://github.com/user-attachments/assets/92a9f1ee-15dc-48a4-8917-1758ca5bae6f" />

### Scaling Data
We use Z Score normalization i.e. divide by standard deviation and subtract by mean.  

### Linear Regression
We can perform linear regression on a dataset, it uses `normal equation` to calculate the parameters theta.  
<img width="1673" height="403" alt="image" src="https://github.com/user-attachments/assets/ec93df97-9099-4d5e-8d5a-385c6c0938c6" />

## Performace Benchmark
We used linear regression on `housing_dataset`.  
Results in **scikit-learn**  
<img width="574" height="195" alt="image" src="https://github.com/user-attachments/assets/96bf24ed-fb87-40ef-8e11-de53654eafd4" />  
Result in our **custom library**  
<img width="681" height="248" alt="image" src="https://github.com/user-attachments/assets/9abd96f5-2c0d-4896-891d-7455e2d98095" />


