# vswift
![R Versions](https://img.shields.io/badge/R-4.2%20%7C%204.3%20%7C%204.4-blue)
[![Test Status](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/gh/donishadsmith/vswift/graph/badge.svg?token=7DYAPU2M0G)](https://codecov.io/gh/donishadsmith/vswift)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)

This R package is a simple, user-friendly tool for train-test splitting and k-fold cross-validation for classification data using various classification algorithms from popular R packages. The 
functions used from packages for each classification algorithms:

  - `lda()` from MASS package for Linear Discriminant Analysis
  - `qda()` from MASS package for Quadratic Discriminant Analysis
  - `glm()` from base package with family = "binomial" for Logistic Regression
  - `svm()` from e1071 package for Support Vector Machines
  - `naive_bayes()` from naivebayes package for Naive Bayes
  - `nnet()` from nnet package for Artificial Neural Network
  - `train.kknn()` from kknn package for K-Nearest Neighbors
  - `rpart()` from rpart package for Decision Trees
  - `randomForest()` from randomForest package for Random Forest
  - `multinom()` from nnet package for Multinomial Regression
  - `xgb.train()` from xgboost package for Gradient Boosting Machines

This package was initially inspired by topepo's [caret](https://github.com/topepo/caret) package.

## Features

- **Versatile Data Splitting**: Perform train-test splits or k-fold cross-validation on your classification data.
- **Support for Popular Algorithms**: Choose from a wide range of classification algorithms such as Linear Discriminant Analysis, Quadratic Discriminant Analysis, Logistic Regression, Support Vector Machines, Naive Bayes, Artificial Neural Networks, K-Nearest Neighbors, Decision Trees, Random Forest, Multinomial Logistic Regression, and Gradient Boosting Machines. Additionally, multiple algorithms can be specified in a single function call.
- **Stratified Sampling Option**: Ensure representative class distribution using stratified sampling based on class proportions.
- **Handling Unseen Categorical Levels**: Automatically exclude observations from the validation/test set with categories not seen during model training. This is particularly helpful for specific algorithms that might throw errors in such cases.
- **Model Saving Capabilities**: Save all models utilized for training and testing.
- **Dataset Saving Options**: Preserve split datasets and folds.
- **Model Creation**: Easily create and save final models.
- **Missing Data Imputation**: Choose from two imputation methods - Bagged Tree Imputation and KNN Imputation. These two methods use the `step_bag_impute()` and `step_knn_impute()` functions from the recipes package, respectively. The recipes package is used to create an imputation model using the training data to predict missing data in the predictors for both  the training data and the validation data. This is done to prevent data leakage. Rows with missing target variables are removed and the target is removed from being a predictor during imputation.
- **Model Creation**: Easily create and save final models.
- **Performance Metrics**: View performance metrics in the console and generate/save plots for key metrics, including overall classification accuracy, as well as f-score, precision, and recall for each class in the target variable across train-test split and k-fold cross-validation.
- **Automatic Numerical Encoding**: Classes within the target variable are automatically numerically encoded for algorithms such as Logistic Regression and Gradient Boosted Models that require numerical inputs for the target variable.
- **Parallel Processing**: Specify the `n_cores` and `future.seed` parameters in `parallel_configs` to specify the number of cores for parallel processing to process multiple folds simultaneously. Only available when cross validation is specified.
- **Minimal Code Requirement**: Access desired information quickly and efficiently with just a few lines of code.

## Installation

### From the "main" branch:

```R
# Install 'remotes' to install packages from Github
install.packages("remotes")

# Install 'vswift' package
remotes::install_github("donishadsmith/vswift/pkg/vswift", ref="main")
 
# Display documentation for the 'vswift' package
help(package = "vswift")
```

### Github release:
```R
# Install 'remotes' to install packages from Github
install.packages("remotes")

# Install 'vswift' package
remotes::install_url("https://github.com/donishadsmith/vswift/releases/download/0.2.3/vswift_0.2.3.tar.gz")

# Display documentation for the 'vswift' package
help(package = "vswift")
```
## Usage

The type of classification algorithm is specified using the `models` parameter in the `classCV()` function.

Acceptable inputs for the `models` parameter includes:

  - "lda" for Linear Discriminant Analysis
  - "qda" for Quadratic Discriminant Analysis
  - "logistic" for Logistic Regression
  - "svm" for Support Vector Machines
  - "naivebayes" for Naive Bayes
  - "ann" for Artificial Neural Network 
  - "knn" for K-Nearest Neighbors
  - "decisiontree" for Decision Trees
  - "randomforest" for Random Forest
  - "multinom" for Multinomial Regression
  - "gbm" for Gradient Boosting Machines

### Using a single model:

```R
# Load the package
library(vswift)

# Perform train-test split and k-fold cross-validation with stratified sampling
results <- classCV(data = iris,
                   target = "Species",
                   models = "lda",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50)
                   )
                   
# Also valid; the target variable can refer to the column index

results <- classCV(data = iris,
                   target = 5,
                   models = "lda",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50))

# Using formula method is also valid 

results <- classCV(formula = Species ~ .,
                   data = iris,
                   models = "lda",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50))
```
`classCV()` produces a vswift object which can be used for custom printing and plotting of performance metrics by using the `print()` and `plot()` functions.
```R
class(results)
```
**Output**
```
[1] "vswift"
```
```R
# Print parameter information and model evaluation metrics
print(results, parameters = TRUE, metrics = TRUE)
```
**Output:**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Linear Discriminant Analysis 

Formula: Species ~ .

Number of Features: 4

Classes: setosa, versicolor, virginica

Training Parameters: list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = NULL, final_model = FALSE)

Missing Data: 0

Effective Sample Size: 150

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = NULL, future.seed = NULL)



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.98 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     0.95      0.97 
virginica             0.95     1.00      0.98 


 Test 
_ _ _ _ 

Classification Accuracy:  0.97 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.91     1.00      0.95 
virginica             1.00     0.90      0.95 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.98 (0.04) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.98 (0.05)       0.96 (0.09)       0.97 (0.07) 
virginica            0.96 (0.08)       0.98 (0.04)       0.97 (0.06) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE, save_plots = TRUE, path = getwd())
```

<details>
  
  <summary>Plots</summary>
  
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/b5436209-d150-40c5-95df-fc0c6c97b02e)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/ee401b50-7afc-436b-88d1-4ac2e5396af7)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/65a499c3-b140-4e0b-becc-515938ced581)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/0fce33c9-1585-411e-8f18-ae0d07297603)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/0c7c0dcc-db28-4600-900d-6578605c9a54)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/38f3be64-72c5-4c30-bec9-84d44c0e6ddd)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/c56b2abd-cd87-426f-a8be-5dc0f41deed2)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/fbc0a9a8-253e-4d8f-b144-b3e0e9c5dec6)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/bbb09b2d-4eec-4188-b363-7470bec9635f)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/809fd1cb-56ab-45d9-b819-716d06689169)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/1f593290-1b4e-4ea4-b2e2-8a9645c4325f)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/aab82a1e-18fe-4506-b409-bef6e5a34218)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/53c686d3-782e-4732-a887-5716d8046a20)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/1bcb2bd0-7874-4c49-83ae-c4110a353f5b)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/38e93e79-3f08-40b5-bc6d-2cad59c4d184)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/3a3de30c-c02a-4b15-9274-d41d80431514)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/5aa8dc51-c03c-4553-94bc-507531897bb5)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/9a5f24b9-7102-4fbb-a250-3fda7a9de9bc)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/abce344d-3f86-4ea4-b59d-c8990c4e9b61)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/8c289855-244d-4b3d-9137-c1873ace31fd)


</details>

The number of predictors can be modified using the `predictors` or `formula` parameters:

```R
# Using knn on iris dataset, using the first, third, and fourth columns as predictors. Also, adding an additional argument, `ks = 5`, which is used in train.kknn() from kknn package

results <- classCV(data = iris,
                   target = "Species",
                   predictors = c("Sepal.Length","Petal.Length","Petal.Width"),
                   models = "knn",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50),
                   ks = 5)

# All configurations below are valid and will produce the same output

args <- list(knn = list(ks = 5))
results <- classCV(data = iris,
                   target = 5,
                   predictors = c(1,3,4),
                   models = "knn",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50),
                   model_params = list(map_args = args))

results <- classCV(formula = Species ~ Sepal.Length + Petal.Length + Petal.Width,
                   data = iris,
                   models = "knn",
                   train_params = list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50),
                   ks = 5)
                   
print(results)
```
**Output**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors 

Formula: Species ~ Sepal.Length + Petal.Length + Petal.Width

Number of Features: 3

Classes: setosa, versicolor, virginica

Training Parameters: list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = list(knn = list(ks = 5)), final_model = FALSE)

Missing Data: 0

Effective Sample Size: 150

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = NULL, future.seed = NULL)



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.97 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.95     0.95      0.95 
virginica             0.95     0.95      0.95 


 Test 
_ _ _ _ 

Classification Accuracy:  0.97 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.91     1.00      0.95 
virginica             1.00     0.90      0.95 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.96 (0.05) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.92 (0.08)       0.96 (0.09)       0.94 (0.08) 
virginica            0.96 (0.09)       0.92 (0.08)       0.94 (0.08) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

Displaying what is contained in the vswift object by converting its class to a list and using R's base `print()` function.

```R
class(results) <- "list"
print(results)
```

<details>
    <summary>Output</summary>

    ```
    $configs
    $configs$formula
    Species ~ Sepal.Length + Petal.Length + Petal.Width
    
    $configs$n_features
    [1] 3
    
    $configs$models
    [1] "knn"
    
    $configs$model_params
    $configs$model_params$map_args
    $configs$model_params$map_args$knn
    $configs$model_params$map_args$knn$ks
    [1] 5
    
    
    
    $configs$model_params$final_model
    [1] FALSE
    
    $configs$model_params$logistic_threshold
    NULL
    
    
    $configs$train_params
    $configs$train_params$split
    [1] 0.8
    
    $configs$train_params$n_folds
    [1] 5
    
    $configs$train_params$stratified
    [1] TRUE
    
    $configs$train_params$random_seed
    [1] 50
    
    $configs$train_params$standardize
    [1] FALSE
    
    $configs$train_params$remove_obs
    [1] FALSE
    
    
    $configs$missing_data
    [1] 0
    
    $configs$effective_sample_size
    [1] 150
    
    $configs$impute_params
    $configs$impute_params$method
    NULL
    
    $configs$impute_params$args
    NULL
    
    
    $configs$parallel_configs
    $configs$parallel_configs$n_cores
    NULL
    
    $configs$parallel_configs$future.seed
    NULL
    
    
    $configs$save
    $configs$save$models
    [1] FALSE
    
    $configs$save$data
    [1] FALSE
    
    
    
    $class_summary
    $class_summary$classes
    [1] "setosa"     "versicolor" "virginica" 
    
    $class_summary$proportions
    target_vector
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $class_summary$indices
    $class_summary$indices$setosa
     [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    
    $class_summary$indices$versicolor
     [1]  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94
    [45]  95  96  97  98  99 100
    
    $class_summary$indices$virginica
     [1] 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144
    [45] 145 146 147 148 149 150
    
    
    
    $data_partitions
    $data_partitions$indices
    $data_partitions$indices$split
    $data_partitions$indices$split$train
      [1]  48  11  31  50  46   3   8  16  18  27  21  41  20  37  34   7  28  29  26  10  25  13   2  30  36  15  47  49  35  40  12  42   4   6  22  44  17   5  39  33  69  66  61  70
     [45]  81  74  88  93  91  87  56  63  52  55  73  72  80  97  62  94  84  86  65  99  98  53  57  58  90  51  96  75  60  78  92  59  89  85  79  71 130 128 131 133 115 150 124 144
     [89] 125 123 110 138 119 101 132 111 143 112 145 139 104 102 121 140 127 105 136 135 103 122 109 141 120 117 113 107 126 118 148 114
    
    $data_partitions$indices$split$test
     [1]  19   1  24  45   9  43  32  38  23  14  83  67  54  64  82  95  68  76  77 100 134 146 116 149 106 147 142 129 108 137
    
    
    $data_partitions$indices$cv
    $data_partitions$indices$cv$fold1
     [1]  48  11  31  50  46   3   8  16  18  27  71  77  70  87  84  57  78  79  76  60 132 113 134 125 148 131 110 143 107 121
    
    $data_partitions$indices$cv$fold2
     [1]  23   5  29   7  10  44  21  47  33  22  98  51  68  72  67  73 100  63  74  75 139 129 147 146 106 116 102 105 115 128
    
    $data_partitions$indices$cv$fold3
     [1]  34  37  40  35  20   2  38  26  28  19  94  99  54  59  61  58  52  53  88  96 118 124 109 141 137 140 127 104 117 103
    
    $data_partitions$indices$cv$fold4
     [1]   4   9  25  49   6  36  30  12   1  14  83  93  82  66  62  56  55  97  80  91 112 108 126 145 114 150 130 111 135 119
    
    $data_partitions$indices$cv$fold5
     [1]  43  39  13  42  41  32  24  17  15  45  95  85  90  69  92  64  89  81  86  65 142 123 120 122 133 149 144 138 101 136
    
    
    
    $data_partitions$proportions
    $data_partitions$proportions$split
    $data_partitions$proportions$split$train
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $data_partitions$proportions$split$test
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    
    $data_partitions$proportions$cv
    $data_partitions$proportions$cv$fold1
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $data_partitions$proportions$cv$fold2
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $data_partitions$proportions$cv$fold3
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $data_partitions$proportions$cv$fold4
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $data_partitions$proportions$cv$fold5
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    
    
    
    $metrics
    $metrics$knn
    $metrics$knn$split
           Set Classification Accuracy Class: setosa Precision Class: setosa Recall Class: setosa F-Score Class: versicolor Precision Class: versicolor Recall Class: versicolor F-Score
    1 Training               0.9666667                       1                    1                     1                   0.9500000                     0.95                  0.950000
    2     Test               0.9666667                       1                    1                     1                   0.9090909                     1.00                  0.952381
      Class: virginica Precision Class: virginica Recall Class: virginica F-Score
    1                       0.95                    0.95                0.9500000
    2                       1.00                    0.90                0.9473684
    
    $metrics$knn$cv
                        Fold Classification Accuracy Class: setosa Precision Class: setosa Recall Class: setosa F-Score Class: versicolor Precision Class: versicolor Recall
    1                 Fold 1              0.86666667                       1                    1                     1                  0.80000000               0.80000000
    2                 Fold 2              0.96666667                       1                    1                     1                  0.90909091               1.00000000
    3                 Fold 3              1.00000000                       1                    1                     1                  1.00000000               1.00000000
    4                 Fold 4              1.00000000                       1                    1                     1                  1.00000000               1.00000000
    5                 Fold 5              0.96666667                       1                    1                     1                  0.90909091               1.00000000
    6               Mean CV:              0.96000000                       1                    1                     1                  0.92363636               0.96000000
    7 Standard Deviation CV:              0.05477226                       0                    0                     0                  0.08272228               0.08944272
    8     Standard Error CV:              0.02449490                       0                    0                     0                  0.03699453               0.04000000
      Class: versicolor F-Score Class: virginica Precision Class: virginica Recall Class: virginica F-Score
    1                0.80000000                 0.80000000              0.80000000               0.80000000
    2                0.95238095                 1.00000000              0.90000000               0.94736842
    3                1.00000000                 1.00000000              1.00000000               1.00000000
    4                1.00000000                 1.00000000              1.00000000               1.00000000
    5                0.95238095                 1.00000000              0.90000000               0.94736842
    6                0.94095238                 0.96000000              0.92000000               0.93894737
    7                0.08231349                 0.08944272              0.08366600               0.08201074
    8                0.03681171                 0.04000000              0.03741657               0.03667632
    ```
</details>

### Using multiple models with parallel processing 

*Note*: This example uses the [internet advertisement data from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/51/internet+advertisements).

```R
# Set url for interet advertisement data from UCI Machine Learning Repository. This data has 3,278 instances and 1558 attributes. 

url <- "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip"

# Set file destination

dest_file <- file.path(getwd(),"ad.zip")

# Download zip file

download.file(url,dest_file)

# Unzip file

unzip(zipfile = dest_file , files = "ad.data")

# Read data

ad_data <- read.csv("ad.data")

# Load in vswift

library(vswift)

# Create arguments variable to tune parameters for multiple models
args <- list("knn" = list(ks = 5), 
             "gbm" = list(params = list(booster = "gbtree", objective = "multi:softmax",
                                        lambda = 0.0003, alpha = 0.0003, num_class = 2, eta = 0.8,
                                        max_depth = 6), nrounds = 10))

print("Without Parallel Processing:")

# Obtain new start time 

start <- proc.time()

# Run the same model without parallel processing 

results <- classCV(data = ad_data,
                   target = "ad.",
                   models = c("knn","svm","decisiontree","gbm"),
                   train_params = list(split = 0.8, n_folds = 5, random_seed = 50),
                   model_params = list(map_args = args)
                   )

# Get end time 
end <- proc.time() - start

# Print time
print(end)

print("Parallel Processing:")

# Adjust maximum object size that can be passed to workers during parallel processing; ~1.2 gb
options(future.globals.maxSize = 1200 * 1024^2)

# Obtain start time
start_par <- proc.time()

# Run model using parallel processing with 4 cores
results <- classCV(data = ad_data,
                   target = "ad.",
                   models = c("knn","svm","decisiontree","gbm"),
                   train_params = list(split = 0.8, n_folds = 5, random_seed = 50),
                   model_params = list(map_args = args),
                   parallel_configs = list(n_cores = 4, future.seed = 100)
                   )

# Obtain end time

end_par <- proc.time() - start_par

# Print time
print(end_par)
```
**Output:**
```
[1] "Without Parallel Processing:"

Warning message:
In .create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
 336.93    1.97  344.62 

[1] "Parallel Processing:"

Warning message:
In .create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
   1.48    9.16  188.28
```

```R
# Print parameter information and model evaluation metrics; If number of features > 20, the tartget replaces the formula
print(results, models = c("gbm", "knn"))
```

**Output:**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Gradient Boosted Machine 

Target: ad.

Number of Features: 1558

Classes: ad., nonad.

Training Parameters: list(split = 0.8, n_folds = 5, random_seed = 50, stratified = FALSE, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = list(gbm = list(params = list(booster = "gbtree", objective = "multi:softmax", lambda = 3e-04, alpha = 3e-04, num_class = 2, eta = 0.8, max_depth = 6), nrounds = 10)), final_model = FALSE)

Missing Data: 0

Effective Sample Size: 3278

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = 4, future.seed = 100)



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.99 

Class:       Precision:  Recall:  F-Score:

ad.               0.99     0.96      0.97 
nonad.            0.99     1.00      1.00 


 Test 
_ _ _ _ 

Classification Accuracy:  0.98 

Class:       Precision:  Recall:  F-Score:

ad.               0.94     0.89      0.92 
nonad.            0.98     0.99      0.99 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.98 (0.01) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.95 (0.02)       0.88 (0.04)       0.91 (0.02) 
nonad.           0.98 (0.01)       0.99 (0.00)       0.99 (0.00) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors 

Target: ad.

Number of Features: 1558

Classes: ad., nonad.

Training Parameters: list(split = 0.8, n_folds = 5, random_seed = 50, stratified = FALSE, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = list(knn = list(ks = 5)), final_model = FALSE)

Missing Data: 0

Effective Sample Size: 3278

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = 4, future.seed = 100)



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  1.00 

Class:       Precision:  Recall:  F-Score:

ad.               1.00     0.99      1.00 
nonad.            1.00     1.00      1.00 


 Test 
_ _ _ _ 

Classification Accuracy:  0.96 

Class:       Precision:  Recall:  F-Score:

ad.               0.89     0.80      0.84 
nonad.            0.97     0.98      0.98 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.93 (0.01) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.71 (0.07)       0.82 (0.01)       0.76 (0.04) 
nonad.           0.97 (0.00)       0.95 (0.02)       0.96 (0.01) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

```

```R
# Plot results

plot(results, models = "gbm" , save_plots = TRUE,
     class_names = "ad.", metrics = c("precision", "recall"))
```

<details>
  
  <summary>Plots</summary>

  ![image](https://github.com/user-attachments/assets/8c9e9a11-92bb-40e3-b104-c5aa5fd5e1a2)
  ![image](https://github.com/user-attachments/assets/db8bd8e6-5ae7-4410-8833-7783b041b31e)
  ![image](https://github.com/user-attachments/assets/3d0f08e1-6967-4c12-be6a-95126161bb0f)
  ![image](https://github.com/user-attachments/assets/a6edc7c9-c925-4a5b-83f1-75432a3f62aa)

</details>
