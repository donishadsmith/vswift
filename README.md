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
                                split = 0.7,
                                n_folds = 5,
                                model_type = "logistic",
                                stratified = TRUE)

# Print parameter information and model evaluation metrics
print(results, parameters = TRUE, metrics = TRUE)

# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE)
```
