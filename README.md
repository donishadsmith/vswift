# vswift
This R package is a simple, user-friendly tool for train-test splitting and k-fold cross-validation of classification data using various classification algorithms from popular R packages. The 
functions used from packages for each classification algorithms:

  - lda() from MASS package for Linear Discriminant Analysis
  - qda() from MASS package for Quadratic Discriminant Analysis
  - glm() from base package with family = "binomial" for Logistic Regression
  - svm() from e1071 package for Support Vector Machines
  - naive_bayes() from naivebayes package for Naive Bayes
  - nnet() from nnet package for Artificial Neural Network
  - train.kknn() from kknn package for K-Nearest Neighbors
  - rpart() from rpart package for Decision Trees
  - randomForest() from randomForest package for Random Forest
  - multinom() from nnet package for Multinomial Regression
  - xgb.train() from xgboost package for Gradient Boosting Machines
  
This package is currently in beta, but it's functional and ready for use. Additionally, I will expand this package to deal with regression outputs in the future.


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
- **Automatic Numerical Encoding**: Classes within the target variable are automaically numerically encoded for algorithms such as Logistic Rgression adn Gradient Boosted Models that require numerical inputs for the target variable.
- **Parallel Processing**: Use the `n_cores` parameter to specify the number of cores for parallel processing.
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
                   random_seed = 123)
                   
class(results)
```
**Output**
```
[1] "vswift"
```
`classCV()` produces a vswift object which can be used for custom printing and plotting of performance metrics by using the `print()` and `plot()` functions.

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
plot(results, split = TRUE, cv = TRUE, save_plots = TRUE)
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

Displaying what is contained in the vswift object by converting its class to a list and using R's base `print()` function.

```R
class(results) <- "list"
print(results)
```

**Output**
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

### Using multiple models with parallel processing 

*Note*: This example uses the internet advertisement (data)[https://archive.ics.uci.edu/dataset/51/internet+advertisements] from the UCI Machine Learning Repository.

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

print("Parallel Processing:")
# Obtain start time
start_par <- proc.time()

# Run model using parallel processing with 4 cores
result <- classCV(data = ad_data, target = "ad.", split = 0.8, n_folds = 5, model_type = c("knn","svm","decisiontree","gbm"), mod_args = args, n_cores = 4)

# Obtain end time

end_par <- proc.time() - start_par

# Print time
print(end_par)

print("Without Parallel Processing:")

# Obtain new start time 

start <- proc.time()

# Run the same model without parallel processing 

result <- classCV(data = ad_data, target = "ad.", split = 0.8, n_folds = 5, model_type = c("knn","svm","decisiontree","gbm"), mod_args = args)

# Get end time 
end <- proc.time() - start

# Print time
print(end)
```
**Output:**
```
[1] "Parallel Processing:"

Warning message:
In vswift:::.create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: ad. = 0, nonad. = 1

> print(end_par)
   user  system elapsed 
   9.55   16.47  173.66 

> print("Without Parallel Processing:")
[1] "Without Parallel Processing:"

Warning message:
In vswift:::.create_dictionary(preprocessed_data = preprocessed_data,  :
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
 318.85    2.28  331.50 
```

```R
# Print parameter information and model evaluation metrics
print(result, model_type = c("gbm", "knn"))
```

**Output:**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Gradient Boosted Machine

Number of Predictors: 1558

Classes: ad., nonad.

Fold size: 5

Split: 0.8

Stratified Sampling: FALSE

Random Seed: 123

missForest Arguments: 

Missing Data: 0

Sample Size: 3278

Additional Arguments: params = list("objective = multi:softprob", "num_class = 2", "eta = 0.3", "max_depth = 6"), nrounds = 10



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  0.98 

Class:       Precision:  Recall:  F-Score:

ad.               0.98     0.90      0.94 
nonad.            0.98     1.00      0.99 


 Test 
_ _ _ _ 

Classication Accuracy:  0.97 

Class:       Precision:  Recall:  F-Score:

ad.               0.98     0.83      0.90 
nonad.            0.97     1.00      0.98 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.97 (0.00) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.95 (0.02)       0.84 (0.02)       0.89 (0.02) 
nonad.           0.97 (0.00)       0.99 (0.00)       0.98 (0.00) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors

Number of Predictors: 1558

Classes: ad., nonad.

Fold size: 5

Split: 0.8

Stratified Sampling: FALSE

Random Seed: 123

missForest Arguments: 

Missing Data: 0

Sample Size: 3278

Additional Arguments: ks = 5



 Training 
_ _ _ _ _ _ _ _ 

Classication Accuracy:  1.00 

Class:       Precision:  Recall:  F-Score:

ad.               1.00     0.99      1.00 
nonad.            1.00     1.00      1.00 


 Test 
_ _ _ _ 

Classication Accuracy:  0.95 

Class:       Precision:  Recall:  F-Score:

ad.               0.86     0.79      0.82 
nonad.            0.96     0.98      0.97 


 K-fold CV 
_ _ _ _ _ _ _ _ _ 

Average Classication Accuracy:  0.92 (0.01) 

Class:       Average Precision:  Average Recall:  Average F-score:

ad.              0.70 (0.06)       0.81 (0.05)       0.75 (0.02) 
nonad.           0.97 (0.01)       0.94 (0.02)       0.95 (0.01) 


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot results

plot(result, model_type = "gbm" , save_plots = TRUE)
```

<details>
  
  <summary>Plots</summary>
  ![Gradient Boosted Machine_train_test_classification_accuracy](https://github.com/donishadsmith/vswift/assets/112973674/fca81609-138d-40e4-b8ba-096b5e2b720d)


</details>
