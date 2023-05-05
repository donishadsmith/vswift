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
# Install 'devtools' to install packages from Github
install.packages("devtools")
# Install 'vswift' package
devtools::install_github(repo = "donishadsmith/vswift", subdir = "pkg/vswift")
# Display documentation for the 'vswift' package
help(package = "vswift")
```
## Usage

```R
# Load the package
library(vswift)

# Perform train-test split and k-fold cross-validation with stratified sampling
results <- categorical_cv_split(data = iris,
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

Average Classication Accuracy:  0.98 (0.01) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.98 (0.02)       0.96 (0.02)       0.97 (0.01) 
virginica            0.96 (0.02)       0.98 (0.02)       0.97 (0.01) 
```
```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE)
```
![image](https://user-images.githubusercontent.com/112973674/236352770-f4264988-099e-459d-ad8c-278624d67ecf.png)
![image](https://user-images.githubusercontent.com/112973674/236352801-d1754848-12e8-4be2-901e-a808363ff530.png)
![image](https://user-images.githubusercontent.com/112973674/236352819-7999d88b-a061-468e-b81d-6be426eebc99.png)
![image](https://user-images.githubusercontent.com/112973674/236352858-ebabd2bb-87c7-4c17-8eee-328ff76c85d3.png)
![image](https://user-images.githubusercontent.com/112973674/236352879-181cf86f-fbb6-47df-a1d2-b7464ab33d04.png)
![image](https://user-images.githubusercontent.com/112973674/236352895-ef4b31f8-90f0-4490-bee9-d502a4693be9.png)
![image](https://user-images.githubusercontent.com/112973674/236352907-d38a93aa-a650-4a9c-ac2a-7c4d6c31551b.png)
![image](https://user-images.githubusercontent.com/112973674/236352928-6a42bb74-d827-4089-8817-4e00ce0b5097.png)
![image](https://user-images.githubusercontent.com/112973674/236352948-4c03edff-72b9-4c3c-904b-8f7c55076ea4.png)
![image](https://user-images.githubusercontent.com/112973674/236352967-a1f96419-6687-497f-9213-af31358413c0.png)
![image](https://user-images.githubusercontent.com/112973674/236352975-4b8f67f8-e56c-4752-8cb9-e1704143b5e6.png)
![image](https://user-images.githubusercontent.com/112973674/236352990-886aadd4-a43c-4604-a846-f35ad05400c4.png)
![image](https://user-images.githubusercontent.com/112973674/236353008-74f28447-87d4-4f07-903f-d43ebfb009d2.png)
