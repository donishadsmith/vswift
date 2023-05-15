# vswift
This R package is a simple, user-friendly tool for train-test splitting and k-fold cross-validation of classification data using various classification algorithms. 

This package is currently in beta, but it's functional and ready for use.


## Features

- **Versatile Data Splitting**: Perform train-test splits or k-fold cross-validation on your classification data.
- **Support for Popular Algorithms**: Choose from a wide range of classification algorithms such as Linear Discriminant Analysis, Quadratic Discriminant Analysis, Logistic Regression, Support Vector Machines, Naive Bayes, Artificial Neural Networks, K-Nearest Neighbors, Decision Trees, Random Forest, Multinomial Logistic Regression, and Gradient Boosting Machines.
- **Stratified Sampling Option**: Ensure representative class distribution by using stratified sampling based on class proportions.
- **Missing Data Imputation**: Impute missing values with Simple Imputation (mean, median, mode) or Random Forest Imputation techniques.
- **Handling Unseen Categorical Levels**: Automatically exclude observations from the validation/test set that have categories not seen during model training. This is particularly helpful for certain algorithms that might throw errors in such cases.
- **Model Saving Capabilities**: Save all models utilized for training and testing.
- **Dataset Saving Options**: Preserve split datasets and folds.
- **Model Creation**: Easily create and save final models.
- **Class Distribution Information**: Obtain information on target class distribution for each training and testing split, as well as within each k-fold, using the classCV() function. The output is a vswift list object, provided stratified sampling is specified.
- **Performance Metrics**: View performance metrics in the console and generate/save plots for key metrics including overall classification accuracy, as well as f-score, precision, and recall for each class in the target variable across train-test split and k-fold cross-validation.
- **Minimal Code Requirement**: Access desired information quickly and efficiently with just a few lines of code.

## Installation

To install and use vswift:

```R
# Install 'devtools' to install packages from Github
install.packages("devtools")

# Install 'vswift' package
devtools::install_github(repo = "donishadsmith/vswift", subdir = "pkg/vswift")

# Display documentation for the 'vswift' package
help(package = "vswift")
```
## Usage

Example using a single model:

```R
# Load the package
library(vswift)

# Perform train-test split and k-fold cross-validation with stratified sampling
results <- classCV(data = iris,
                   target = "Species",
                   split = 0.8,
                   n_folds = 5,
                   model_type = "lda",
                   stratified = TRUE,
                   random_seed = 123)                              
```
```R
# Print parameter information and model evaluation metrics
print(results, parameters = TRUE, metrics = TRUE)
```
**Output:**
```
Model Type: lda

Predictors: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 5

Split: 0.8

Stratified Sampling: TRUE

Random Seed: 123

Missing Data: 0

Sample Size: 150

Additional Arguments: 



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  0.98 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.97     0.95      0.96 
virginica             0.95     0.98      0.96 


 Test 
_ _ _ _ 

Classication Accuracy:  1.00 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     1.00      1.00 
virginica             1.00     1.00      1.00 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.98 (0.02) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.98 (0.04)       0.96 (0.05)       0.97 (0.03) 
virginica            0.96 (0.05)       0.98 (0.04)       0.97 (0.03) 
```
```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE, save_plots = FALSE)
```

<details>
  
  <summary>Plots</summary>
  
  ![image](https://user-images.githubusercontent.com/112973674/236356074-7f420bc3-63fd-4407-9dc7-4ed09506886c.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356083-f59ebafc-e5a4-4dab-a696-de5a6ae723be.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356088-fe71f5a3-ecfa-4934-9049-13305ce5d56e.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356101-8eccba78-b0be-4473-a822-61eb00edc8d9.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356111-1cf184ba-6ef3-41a4-8c95-5f98902c72ee.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356127-fb8c7da4-762c-4164-a8f0-0496e66c8c04.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356144-33e15f57-ed6a-4be0-a798-e5fbef815e2d.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356158-0afc4f51-7a62-4a28-bd60-07c2a632d311.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356175-d45d32af-115c-48cc-8503-275197a48b61.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356220-998d8e21-0640-444f-8f0b-f3e8133bfe82.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356233-3f5fd69d-d3a3-4d1e-ab51-340e5351db86.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356246-da9dc81e-ae80-4353-8261-a31d9854c9b5.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356256-db6dab3c-7c51-46ad-a6f8-a23cc845ea63.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356275-bafd982f-f6d8-4368-a240-202277ecdc09.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356283-bf0e47dc-f8ac-41fe-bae7-49dbd5785d9e.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356294-933e0eda-e7c6-47d8-8017-a817b4394a01.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356306-b101c0a4-049e-45f1-bf70-371968d90b06.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356316-6bbad7cd-ddf3-460a-8c41-e4198cdb58f6.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356328-45cdbd5d-a88b-4191-92b7-8503d44fb036.png)
  ![image](https://user-images.githubusercontent.com/112973674/236356341-2af76050-18bf-4d21-8504-c598176f8269.png)

</details>
  
Example using multiple models:

```R
args <- list("knn" = list(ks = 3), "ann" = list(size = 10), "gbm" = list(params = list(objective =         "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10))

result <- classCV(data = iris, target = 5, split = 0.8, model_type = c("decisiontree","gbm","knn", "ann","svm"), 
n_folds = 3, mod_args = args, save_data = T, save_models = T, remove_obs = T, stratified = T)
```
**Output:**
```
 # weights:  83
initial  value 179.463681 
iter  10 value 20.460222
iter  20 value 5.942445
iter  30 value 3.681145
iter  40 value 1.682037
iter  50 value 0.005365
final  value 0.000066 
converged
# weights:  83
initial  value 154.009655 
iter  10 value 27.876365
iter  20 value 6.000785
iter  30 value 4.720259
iter  40 value 4.701999
iter  50 value 4.701278
final  value 4.701277 
converged
# weights:  83
initial  value 117.207968 
iter  10 value 29.546902
iter  20 value 0.207537
iter  30 value 0.001366
final  value 0.000079 
converged
# weights:  83
initial  value 126.120778 
iter  10 value 22.654274
iter  20 value 4.855594
iter  30 value 2.928667
iter  40 value 1.925315
iter  50 value 1.446492
iter  60 value 1.394366
iter  70 value 1.388966
iter  80 value 1.386683
iter  90 value 1.386305
iter 100 value 1.386295
final  value 1.386295 
stopped after 100 iterations
# weights:  83
initial  value 134.405440 
iter  10 value 6.801684
iter  20 value 4.612546
iter  30 value 3.713089
iter  40 value 3.357661
iter  50 value 3.248110
iter  60 value 2.771384
iter  70 value 0.298555
iter  80 value 0.010594
iter  90 value 0.002036
iter 100 value 0.001188
final  value 0.001188 
stopped after 100 iterations
Warning message:
In vswift:::.create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: setosa = 0, versicolor = 1, virginica = 2
```
```R
# Print parameter information and model evaluation metrics
print(result, model_type = c("decisiontree", "ann", "knn"))
```
**Output:**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Decision Tree

Predictors: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 3

Split: 0.8

Stratified Sampling: TRUE

missForest Arguments: 

Missing Data: 0

Sample Size: 150

Additional Arguments: 



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  0.97 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     0.90      0.95 
virginica             0.91     1.00      0.95 


 Test 
_ _ _ _ 

Classication Accuracy:  0.90 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.89     0.80      0.84 
virginica             0.82     0.90      0.86 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.94 (0.02) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.89 (0.05)       0.94 (0.00)       0.91 (0.02) 
virginica            0.94 (0.00)       0.88 (0.06)       0.91 (0.03) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Neural Network

Predictors: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 3

Split: 0.8

Stratified Sampling: TRUE

missForest Arguments: 

Missing Data: 0

Sample Size: 150

Additional Arguments: size = 10



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  1.00 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     1.00      1.00 
virginica             1.00     1.00      1.00 


 Test 
_ _ _ _ 

Classication Accuracy:  0.93 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     0.80      0.89 
virginica             0.83     1.00      0.91 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.94 (0.04) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       0.98 (0.04)       0.99 (0.02) 
versicolor           0.92 (0.02)       0.90 (0.13)       0.90 (0.07) 
virginica            0.91 (0.10)       0.94 (0.06)       0.92 (0.03) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors

Predictors: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 3

Split: 0.8

Stratified Sampling: TRUE

missForest Arguments: 

Missing Data: 0

Sample Size: 150

Additional Arguments: ks = 3



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  1.00 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     1.00      1.00 
virginica             1.00     1.00      1.00 


 Test 
_ _ _ _ 

Classication Accuracy:  0.90 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.89     0.80      0.84 
virginica             0.82     0.90      0.86 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.94 (0.02) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.92 (0.07)       0.90 (0.04)       0.91 (0.03) 
virginica            0.90 (0.03)       0.92 (0.07)       0.91 (0.03) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```
