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
./linear_regression ../data/LinearRegression/california_housing.csv 9 0
```
Here, `../data/LinearRegression/california_housing.csv` is path to dataset and `9` is the column where target variable is located and `0` clarifies that the target columnn in numerical.  
See `data` directory to see various demo datasets. 

## Features Implemented and Demonstration
### CSV Parser
The `load_csv()` function parses a CSV file and stores the numberical data while elegantly formatting it to be displayed on the console using `.print_dataset()` function.  

<img width="1753" height="229" alt="image" src="https://github.com/user-attachments/assets/8b8d4f07-80e2-4755-b21b-ec825bcbacd6" />  

### Splitting data
We can now use `test_train_splie()` to split testing and training data.  
<img width="876" height="243" alt="image" src="https://github.com/user-attachments/assets/92a9f1ee-15dc-48a4-8917-1758ca5bae6f" />

### Scaling Data
We use Z Score normalization i.e. divide by standard deviation and subtract by mean.  

### Linear Regression
We can perform linear regression on a dataset, it uses `normal equation` to calculate the parameters theta.  
<img width="1673" height="403" alt="image" src="https://github.com/user-attachments/assets/ec93df97-9099-4d5e-8d5a-385c6c0938c6" />

### Logistic Regression
We can perform logistic regression on a dataset, it uses `gradient ascent` and `sigmoid function` to find the parameters
<img width="1720" height="562" alt="image" src="https://github.com/user-attachments/assets/77a3690a-ac1e-452c-9f31-07d319ab2792" />

### Naive Bayes
We have also impleamented `Gaussian Naive Bayes` used for classification.  


## Performace Benchmark

### Linear Regression
We used linear regression on `housing_dataset`.  
Results in **scikit-learn**  
<img width="574" height="195" alt="image" src="https://github.com/user-attachments/assets/96bf24ed-fb87-40ef-8e11-de53654eafd4" />  
Result in our **custom library**  
<img width="681" height="248" alt="image" src="https://github.com/user-attachments/assets/9abd96f5-2c0d-4896-891d-7455e2d98095" />  

### Logistic Regression
We used logistic regression on `breast cancer` dataset  
Results in **scikit-learn**  
<img width="235" height="196" alt="image" src="https://github.com/user-attachments/assets/8fd6f67f-d5a3-4261-8897-51a3e7c04676" />  
Results in **custom library**  
<img width="1720" height="562" alt="image" src="https://github.com/user-attachments/assets/df0bb275-814b-4244-b982-85610cad50dd" />  

## Naive Bayes
We use naive bayes on `breast cancer` dataset  
<img width="238" height="224" alt="image" src="https://github.com/user-attachments/assets/3c01446d-b0e4-47e4-a0ba-79c26a8b206d" />  
Comparing it to performance in sklearn.  
<img width="386" height="186" alt="image" src="https://github.com/user-attachments/assets/48eca82e-9091-4f82-ba3d-e813c77d375d" />  





