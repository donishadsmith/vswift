# vswift
[![R Versions](https://img.shields.io/badge/R-4.2%20%7C%204.3%20%7C%204.4-blue)](https://github.com/donishadsmith/vswift)
[![Test Status](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/gh/donishadsmith/vswift/graph/badge.svg?token=7DYAPU2M0G)](https://codecov.io/gh/donishadsmith/vswift)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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

**Still in beta but stable.**
## Features

- **Versatile Data Splitting**: Perform train-test splits or k-fold cross-validation on your classification data.
- **Support for Popular Algorithms**: Choose from a wide range of classification algorithms such as Linear Discriminant Analysis, Quadratic Discriminant Analysis, Logistic Regression, Support Vector Machines, Naive Bayes, Artificial Neural Networks, K-Nearest Neighbors, Decision Trees, Random Forest, Multinomial Logistic Regression, and Gradient Boosting Machines. Additionally, multiple algorithms can be specified in a single function call.
- **Stratified Sampling Option**: Ensure representative class distribution using stratified sampling based on class proportions. This package uses [custom code](https://github.com/donishadsmith/vswift/blob/3572dc7eb4fadb22ea83d6d3eb5dc6fa9de1bf1c/pkg/vswift/R/stratified_sampling.R#L1C1-L102) to accomplish this.
- **Handling Unseen Categorical Levels**: Automatically exclude observations from the validation/test set with categories not seen during model training. This is particularly helpful for specific algorithms that might throw errors in such cases. [Link to code.](https://github.com/donishadsmith/vswift/blob/3572dc7eb4fadb22ea83d6d3eb5dc6fa9de1bf1c/pkg/vswift/R/validation_internals.R#L107-L138)
- **Model Saving Capabilities**: Save all models utilized for training and testing.
- **Dataset Saving Options**: Preserve split datasets and folds.
- **Model Creation**: Easily create and save final models.
- **Missing Data Imputation**: Choose from two imputation methods - Bagged Tree Imputation and KNN Imputation. These two methods use the `step_bag_impute()` and `step_knn_impute()` functions from the recipes package, respectively. The recipes package is used to create an imputation model using the training data to predict missing data in the training data and the validation data. This is done to prevent data leakage. Rows with missing target variables are removed. If predictors are specified using the `predictors =` parameter in the `classCV` function, only those predictors are imputed. Link to relevant code [here](https://github.com/donishadsmith/vswift/blob/d69606afc391a011a8e2bd9664a13b072c12a509/pkg/vswift/R/preprocess.R#L220-L406) and [here](https://github.com/donishadsmith/vswift/blob/d69606afc391a011a8e2bd9664a13b072c12a509/pkg/vswift/R/classCV.R#L265-L281).
- **Model Creation**: Easily create and save final models.
- **Performance Metrics**: View performance metrics in the console and generate/save plots for key metrics, including overall classification accuracy, as well as f-score, precision, and recall for each class in the target variable across train-test split and k-fold cross-validation. Link to relevant code are [here](https://github.com/donishadsmith/vswift/blob/83b482b18609799a3a31754a826f9579223db26b/pkg/vswift/R/validation_internals.R#L141-L275), [here](https://github.com/donishadsmith/vswift/blob/2d729dfaab55f3649369b202a393521bd42164b1/pkg/vswift/R/classCV.R#L300-L318), and [here](https://github.com/donishadsmith/vswift/blob/83b482b18609799a3a31754a826f9579223db26b/pkg/vswift/R/classCV.R#L319-L331) to show how the metrics are calculated.
- **Automatic Numerical Encoding**: Classes within the target variable are automatically numerically encoded for algorithms such as Logistic Regression and Gradient Boosted Models that require numerical inputs for the target variable.
- **Parallel Processing**: Use the `n_cores` parameter to specify the number of cores for parallel processing to process multiple folds simultaneously. Only available when cross validation is specified.  Link to relevant code are [here](https://github.com/donishadsmith/vswift/blob/83b482b18609799a3a31754a826f9579223db26b/pkg/vswift/R/classCV.R#L289C9-L317) and [here](https://github.com/donishadsmith/vswift/blob/83b482b18609799a3a31754a826f9579223db26b/pkg/vswift/R/validation_internals.R#L277-L318)
- **Minimal Code Requirement**: Access desired information quickly and efficiently with just a few lines of code.

## Installation

To install and use vswift:

```R
# Install 'remotes' to install packages from Github
install.packages("remotes")

# Install 'vswift' package
remotes::install_url("https://github.com/donishadsmith/vswift/releases/download/0.1.2/vswift_0.1.2.tar.gz")

# Display documentation for the 'vswift' package
help(package = "vswift")
```
## Usage

The type of classification algorithm is specified using the `model_type` parameter in the `classCV()` function.

Acceptable inputs for the `model_type` parameter includes:

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
                   split = 0.8,
                   n_folds = 5,
                   model_type = "lda",
                   stratified = TRUE,
                   random_seed = 123,
                   standardize = TRUE)
                   
# Also valid; the target variable can refer to the column index

results <- classCV(data = iris,
                   target = 5,
                   split = 0.8,
                   n_folds = 5,
                   model_type = "lda",
                   stratified = TRUE,
                   random_seed = 123,
                   standardize = TRUE)

# Using formula method is also valid 

results <- classCV(formula = Species ~ .,
                   data = iris,
                   split = 0.8,
                   n_folds = 5,
                   model_type = "lda",
                   stratified = TRUE,
                   random_seed = 123,
                   standardize = TRUE)
                
                   
class(results)
```
**Output**
```
[1] "vswift"
```
`classCV()` produces a vswift object which can be used for custom printing and plotting of performance metrics by using the `print()` and `plot()` functions.

```R
print(results$formula)
```
**Output**
```
[1] Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
```

```R
# Print parameter information and model evaluation metrics
print(results, parameters = TRUE, metrics = TRUE)
```
**Output:**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Model: Linear Discriminant Analysis

Predictors: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width

Target: Species

Formula: Species ~ .

Classes: setosa, versicolor, virginica

Fold size: 5

Split: 0.8

Stratified Sampling: TRUE

Random Seed: 123

Missing Data: 0

Sample Size: 150

Additional Arguments: 

Parallel: FALSE



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.98 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.97     0.95      0.96 
virginica             0.95     0.98      0.96 


 Test 
_ _ _ _ 

Classification Accuracy:  1.00 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     1.00      1.00 
virginica             1.00     1.00      1.00 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.98 (0.02) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.98 (0.04)       0.96 (0.05)       0.97 (0.03) 
virginica            0.96 (0.05)       0.98 (0.04)       0.97 (0.03) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE, save_plots = TRUE)
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

The number of predictors can be modified using the `predictors` parameter:

```R
# Using knn on iris dataset, using the first, third, and fourth columns as predictors. Also, adding an additional argument, `ks = 5`, which is used in train.kknn() from kknn package

results <- classCV(data = iris,
                   target = "Species",
                   predictors = c("Sepal.Length","Petal.Length","Petal.Width"),
                   split = 0.8,
                   n_folds = 5,
                   model_type = "knn",
                   stratified = TRUE,
                   random_seed = 123,
                   ks = 5)

# All configurations below are valid and will produce the same output

results <- classCV(data = iris,
                   target = 5,
                   predictors = c(1,3,4),
                   split = 0.8,
                   n_folds = 5,
                   model_type = "knn",
                   stratified = TRUE,
                   random_seed = 123,
                   ks = 5)
                   
results <- classCV(data = iris,
                   target = 5,
                   predictors = c("Sepal.Length","Petal.Length","Petal.Width"),
                   split = 0.8,
                   n_folds = 5,
                   model_type = "knn",
                   stratified = TRUE,
                   random_seed = 123,
                   ks = 5)

results <- classCV(data = iris,
                   target = "Species",
                   predictors = c(1,3,4),
                   split = 0.8,
                   n_folds = 5,
                   model_type = "knn",
                   stratified = TRUE,
                   random_seed = 123,
                   ks = 5)

results <- classCV(formula = Species ~ Sepal.Length + Petal.Length + Petal.Width,
                   data = iris,
                   split = 0.8,
                   n_folds = 5,
                   model_type = "knn",
                   stratified = TRUE,
                   random_seed = 123,
                   ks = 5)
                   
print(results)
```
**Output**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors

Predictors: Sepal.Length, Petal.Length, Petal.Width

Target: Species

Formula: Species ~ Sepal.Length + Petal.Length + Petal.Width

Classes: setosa, versicolor, virginica

Fold size: 5

Split: 0.8

Stratified Sampling: TRUE

Random Seed: 123

Missing Data: 0

Sample Size: 150

Additional Arguments: ks = 5

Parallel: FALSE



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.96 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            0.93     0.95      0.94 
virginica             0.95     0.92      0.94 


 Test 
_ _ _ _ 

Classification Accuracy:  1.00 

Class:           Precision:  Recall:  F-Score:

setosa                1.00     1.00      1.00 
versicolor            1.00     1.00      1.00 
virginica             1.00     1.00      1.00 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.97 (0.04) 

Class:           Average Precision:  Average Recall:  Average F-score:

setosa               1.00 (0.00)       1.00 (0.00)       1.00 (0.00) 
versicolor           0.95 (0.08)       0.96 (0.05)       0.95 (0.06) 
virginica            0.96 (0.06)       0.94 (0.09)       0.95 (0.06) 


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
    $analysis_type
    [1] "classification"
    
    $parameters
    $parameters$predictors
    [1] "Sepal.Length" "Sepal.Width"  "Petal.Length" "Petal.Width" 
    
    $parameters$target
    [1] "Species"
    
    $parameters$model_type
    [1] "lda"
    
    $parameters$split
    [1] 0.8
    
    $parameters$n_folds
    [1] 5
    
    $parameters$stratified
    [1] TRUE
    
    $parameters$random_seed
    [1] 123
    
    $parameters$missing_data
    [1] 0
    
    $parameters$sample_size
    [1] 150
    
    $parameters$additional_arguments
    list()
    
    
    $classes
    $classes$Species
    [1] "setosa"     "versicolor" "virginica" 
    
    
    $formula
    Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
    <environment: 0x000001c00be569e0>
    
    $class_indices
    $class_indices$setosa
     [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    [33] 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
    
    $class_indices$versicolor
     [1]  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74
    [25]  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98
    [49]  99 100
    
    $class_indices$virginica
     [1] 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124
    [25] 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148
    [49] 149 150
    
    
    $class_proportions
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_indices
    $sample_indices$split
    $sample_indices$split$training
      [1]  31  15  14   3  42  43  37  48  25  26  27   5  40  28   9  29   8  41   7  10  36  19   4  45
     [25]  17  11  32  21  12  49  50  13  24  30  33  20  18  46  22  39  63  68  83  77  75  88  71  65
     [49]  91  76  81  66  80  56  58  72  85  90  89  67  84  99  98  52  54 100  55  95  69  70  64  53
     [73]  86  82  94  62  59  73  57  96 142 144 134 110 122 112 120 117 135 140 130 115 124 123 107 129
     [97] 139 137 126 106 114 136 127 147 105 131 116 121 111 104 145 141 150 109 118 102 113 103 108 133
    
    $sample_indices$split$test
     [1]  35  23   6  34   1   2  47  16  44  38  87  79  60  97  78  93  74  61  92  51 143 128 101 138
    [25] 149 148 146 119 132 125
    
    
    $sample_indices$cv
    $sample_indices$cv$`fold 1`
     [1]  31  15  14   3  42  43  37  48  25  26  77  55 100  78  59  79  85  58  76  57 142 109 119 136
    [25] 114 117 143 139 112 115
    
    $sample_indices$cv$`fold 2`
     [1]  39   8  10  11  28  33  49  44  50   7  83  93  88  56  62  66  67  72  51  99 133 131 148 127
    [25] 121 132 138 122 126 137
    
    $sample_indices$cv$`fold 3`
     [1]   9  18  13  34  41  12  23  24  40  27  81  53  60  73  61  89  84  86  92  74 144 140 103 108
    [25] 124 116 120 147 141 107
    
    $sample_indices$cv$`fold 4`
     [1]   4  35   6  19  47  21   2  46  29  32  75  82  90  95  96  94  68  63  70  87 134 145 125 111
    [25] 129 150 149 123 110 130
    
    $sample_indices$cv$`fold 5`
     [1]  45  20  22  38  16  17  30   5  36   1  64  97  80  71  54  52  98  91  65  69 104 128 105 118
    [25] 135 101 113 102 146 106
    
    
    
    $sample_proportions
    $sample_proportions$split
    $sample_proportions$split$training
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_proportions$split$test
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    
    $sample_proportions$cv
    $sample_proportions$cv$`fold 1`
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_proportions$cv$`fold 2`
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_proportions$cv$`fold 3`
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_proportions$cv$`fold 4`
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    $sample_proportions$cv$`fold 5`
    
        setosa versicolor  virginica 
     0.3333333  0.3333333  0.3333333 
    
    
    
    $metrics
    $metrics$lda
    $metrics$lda$split
           Set Classification Accuracy Class: setosa Precision Class: setosa Recall
    1 Training                   0.975                       1                    1
    2     Test                   1.000                       1                    1
      Class: setosa F-Score Class: versicolor Precision Class: versicolor Recall
    1                     1                    0.974359                     0.95
    2                     1                    1.000000                     1.00
      Class: versicolor F-Score Class: virginica Precision Class: virginica Recall
    1                 0.9620253                  0.9512195                   0.975
    2                 1.0000000                  1.0000000                   1.000
      Class: virginica F-Score
    1                 0.962963
    2                 1.000000
    
    $metrics$lda$cv
                        Fold Classification Accuracy Class: setosa Precision Class: setosa Recall
    1                 Fold 1             1.000000000                       1                    1
    2                 Fold 2             1.000000000                       1                    1
    3                 Fold 3             0.966666667                       1                    1
    4                 Fold 4             0.966666667                       1                    1
    5                 Fold 5             0.966666667                       1                    1
    6               Mean CV:             0.980000000                       1                    1
    7 Standard Deviation CV:             0.018257419                       0                    0
    8     Standard Error CV:             0.008164966                       0                    0
      Class: setosa F-Score Class: versicolor Precision Class: versicolor Recall
    1                     1                  1.00000000               1.00000000
    2                     1                  1.00000000               1.00000000
    3                     1                  1.00000000               0.90000000
    4                     1                  0.90909091               1.00000000
    5                     1                  1.00000000               0.90000000
    6                     1                  0.98181818               0.96000000
    7                     0                  0.04065578               0.05477226
    8                     0                  0.01818182               0.02449490
      Class: versicolor F-Score Class: virginica Precision Class: virginica Recall
    1                1.00000000                 1.00000000              1.00000000
    2                1.00000000                 1.00000000              1.00000000
    3                0.94736842                 0.90909091              1.00000000
    4                0.95238095                 1.00000000              0.90000000
    5                0.94736842                 0.90909091              1.00000000
    6                0.96942356                 0.96363636              0.98000000
    7                0.02798726                 0.04979296              0.04472136
    8                0.01251628                 0.02226809              0.02000000
      Class: virginica F-Score
    1               1.00000000
    2               1.00000000
    3               0.95238095
    4               0.94736842
    5               0.95238095
    6               0.97042607
    7               0.02707463
    8               0.01210814
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
args <- list("knn" = list(ks = 5), "gbm" = list(params = list(objective = "multi:softprob",num_class = 2,eta = 0.3,max_depth = 6), nrounds = 10))

print("Without Parallel Processing:")

# Obtain new start time 

start <- proc.time()

# Run the same model without parallel processing 

results <- classCV(data = ad_data, target = "ad.", split = 0.8, n_folds = 5, model_type = c("knn","svm","decisiontree","gbm"), mod_args = args, random_seed = 123)

# Get end time 
end <- proc.time() - start

# Print time
print(end)

print("Parallel Processing:")
# Obtain start time
start_par <- proc.time()

# Run model using parallel processing with 4 cores
results <- classCV(data = ad_data, target = "ad.", split = 0.8, n_folds = 5, model_type = c("knn","svm","decisiontree","gbm"), mod_args = args, n_cores = 4, random_seed = 123)

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
 333.57    1.94  343.02  

[1] "Parallel Processing:"

Warning message:
In .create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
   7.49   16.70  206.94 
```

```R
# Print parameter information and model evaluation metrics
print(results, model_type = c("gbm", "knn"))
```

**Output:**
```
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Gradient Boosted Machine

Number of Predictors: 1558

Target: ad.

Classes: ad., nonad.

Fold size: 5

Split: 0.8

Stratified Sampling: FALSE

Random Seed: 123

Missing Data: 0

Sample Size: 3278

Additional Arguments: params = list("objective = multi:softprob", "num_class = 2", "eta = 0.3", "max_depth = 6"), nrounds = 10

Parallel: TRUE



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  0.98 

Class:       Precision:  Recall:  F-Score:

ad.               0.98     0.90      0.94 
nonad.            0.98     1.00      0.99 


 Test 
_ _ _ _ 

Classification Accuracy:  0.97 

Class:       Precision:  Recall:  F-Score:

ad.               0.98     0.83      0.90 
nonad.            0.97     1.00      0.98 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.97 (0.00) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.95 (0.02)       0.84 (0.02)       0.89 (0.02) 
nonad.           0.97 (0.00)       0.99 (0.00)       0.98 (0.00) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors

Number of Predictors: 1558

Target: ad.

Classes: ad., nonad.

Fold size: 5

Split: 0.8

Stratified Sampling: FALSE

Random Seed: 123

Missing Data: 0

Sample Size: 3278

Additional Arguments: ks = 5

Parallel: TRUE



 Training 
_ _ _ _ _ _ _ _ 

Classification Accuracy:  1.00 

Class:       Precision:  Recall:  F-Score:

ad.               1.00     0.99      1.00 
nonad.            1.00     1.00      1.00 


 Test 
_ _ _ _ 

Classification Accuracy:  0.95 

Class:       Precision:  Recall:  F-Score:

ad.               0.86     0.79      0.82 
nonad.            0.96     0.98      0.97 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.92 (0.01) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.70 (0.06)       0.81 (0.05)       0.75 (0.02) 
nonad.           0.97 (0.01)       0.94 (0.02)       0.95 (0.01) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot results

plot(results, model_type = "gbm" , save_plots = TRUE,
     class_names = "ad.", metrics = c("precision", "recall"))
```

<details>
  
  <summary>Plots</summary>
  
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/a40add51-661e-42f5-861a-7776483352aa)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/945d31db-940d-4feb-9697-cdade79f0fc3)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/e3c6a948-1ef4-4a7a-98fe-8a778fb3aac5)
  ![image](https://github.com/donishadsmith/vswift/assets/112973674/dae5e55f-0bf4-421f-82a2-5844798f23d6)

</details>




