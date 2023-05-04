# vswift
This R package is a user-friendly tool for train-test splitting and k-fold cross-validation of classification data using various classification algorithms. It also provides detailed information about class distribution in train/test splits and k-folds.

This package is currently in its beta testing phase but is functional.


## Features

- Perform train-test splits and/or k-fold cross-validation on classification data.
- Support for various classification algorithms: LDA, QDA, logistic regression, SVM, naive Bayes, ANN, kNN, decision tree, and random forest.
- Option to perform stratified sampling based on class distribution.
- Detailed information about the distribution of target classes in each train/test split and k-fold.
- Easy-to-use functions for printing and plotting model evaluation metrics.
- Minimal code required to access the desired information.

## Installation

To install and use vswift:

```R
install.packages("devtools")

devtools::install_github(repo = "donishadsmith/vswift", subdir = "pkg/vswift")

help(package = "vswift")
```
## Usage

```R
# Load the package
library(vswift)

# Perform train-test split and k-fold cross-validation with stratified sampling
results <- categorical_cv_split(data = my_data,
                                target = "class",
                                split = 0.8,
                                n_folds = 5,
                                model_type = "lda",
                                stratified = TRUE)
                                
```
```R
# Print parameter information and model evaluation metrics
print(results, parameters = TRUE, metrics = TRUE)
```
```
Model Type: lda

Predictors: Sepal.Length,Sepal.Width,Petal.Length,Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 5

Split: 0.8

Stratified Sampling: TRUE

Missing Data: 0

Sample Size: 150

Additional Arguments: 



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  0.98 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     0.95      0.97 
virginica             0.95     1.00      0.98 


 Test 
_ _ _ _ 

Classication Accuracy:  0.97 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.91     1.00      0.95 
virginica             1.00     0.90      0.95 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.98 (0.01) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.98 (0.04)       0.96 (0.05)       0.97 (0.05) 
virginica            0.96 (0.05)       0.98 (0.04)       0.97 (0.04) 
```
```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE)
```
