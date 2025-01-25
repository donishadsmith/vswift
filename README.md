# vswift
![R Versions](https://img.shields.io/badge/R-4.2%20%7C%204.3%20%7C%204.4-blue)
[![Test Status](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/vswift/actions/workflows/testing.yaml)
[![Codecov](https://codecov.io/gh/donishadsmith/vswift/graph/badge.svg?token=7DYAPU2M0G)](https://codecov.io/gh/donishadsmith/vswift)
[![Code Style: styler](https://img.shields.io/badge/Code%20Style-styler-a18595)](https://github.com/r-lib/styler)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)

This R package streamlines the process of train-test splitting and cross-validation for classification tasks,
providing a unified interface to multiple classification algorithms from popular R packages through a single
function call.

## Supported Classification Algorithms
The following classification algorithms are available through their respective R packages:

  - `lda` from MASS package for Linear Discriminant Analysis
  - `qda` from MASS package for Quadratic Discriminant Analysis
  - `glm` from base package with `family = "binomial"` for Unregularized Logistic Regression
  - `glmnet` from `glmnet` package with `family = "binomial"` or `family = "multinomial"`and using `cv.glmnet` to select the optimal lambda for
  Regularized Logistic Regression and Regularized Multinomial Logistic Regression.
  - `svm` from e1071 package for Support Vector Machine
  - `naive_bayes` from naivebayes package for Naive Bayes
  - `nnet` from nnet package for Neural Network
  - `train.kknn` from kknn package for K-Nearest Neighbors
  - `rpart` from rpart package for Decision Trees
  - `randomForest` from randomForest package for Random Forest
  - `multinom` from nnet package for Unregularized Multinomial Logistic Regression
  - `xgb.train` from xgboost package for Extreme Gradient Boosting

## Features

### Data Handling
- **Versatile Data Splitting**: Perform train-test splits or cross-validation on your classification data.
- **Stratified Sampling Option**: Ensure representative class distribution using stratified sampling based on class proportions.
- **Handling Unseen Categorical Levels**: Automatically exclude observations from the validation/test set with categories not seen during model training.

### Model Configuration
- **Support for Popular Algorithms**: Choose from a wide range of classification algorithms. Multiple algorithms can be specified in a single function call.
- **Model Saving Capabilities**: Save all models utilized for training and testing for both train-test splitting and cross-validation.
- **Final Model Creation**: Easily create and save final models for future use.
- **Dataset Saving Options**: Preserve split datasets and folds for reproducibility.

### Data Preprocessing
- **Missing Data Imputation**: Select either Bagged Tree Imputation or KNN Imputation, implemented using the recipes package. Imputation uses only feature data from the training set to prevent leakage.
- **Automatic Numerical Encoding**: Target variable classes are automatically encoded numerically for algorithms requiring numerical inputs.

### Model Evaluation
- **Comprehensive Metrics**: Generate and save performance metrics including classification accuracy, precision, recall, and F1 for each class. For binary classification tasks, produce ROC (Receiver Operating Characteristic) curves and calculate ROC-AUC (Area Under Curve) scores.

### Performance & Efficiency
- **Parallel Processing**: Utilize multi-core processing for cross-validation through the future package, configurable via `n_cores` and `future.seed` keys in the `parallel_configs` parameter.
- **Minimal Code Requirement**: Access all functionality efficiently with just a few lines of code.

## Installation

### From the "main" branch

```R
# Install 'devtools' to install packages from Github
install.packages("devtools")

# Install 'vswift' package
devtools::install_github("donishadsmith/vswift", build_manual = TRUE, build_vignettes = TRUE)
 
# Display documentation for the 'vswift' package
help(package = "vswift")
```

### Github release

```R
# Install 'vswift' package
install.packages(
  "https://github.com/donishadsmith/vswift/releases/download/0.4.0.9006/vswift_0.4.0.9006.tar.gz",
  repos = NULL,
  type = "source"
)

# Display documentation for the 'vswift' package
help(package = "vswift")
```
## Usage

The type of classification algorithm is specified using the `models` parameter in the `classCV` function.

Acceptable inputs for the `models` parameter includes:

  - "lda" for Linear Discriminant Analysis
  - "qda" for Quadratic Discriminant Analysis
  - "logistic" for Unregularized Logistic Regression
  - "regularized_logistic" for Regularized Logistic Regression
  - "svm" for Support Vector Machine
  - "naivebayes" for Naive Bayes
  - "nnet" for Neural Network 
  - "knn" for K-Nearest Neighbors
  - "decisiontree" for Decision Trees
  - "randomforest" for Random Forest
  - "multinom" for Unregularized Multinomial Logistic Regression
  - "regularized_multinomial" for Regularized Multinomial Logistic Regression
  - "xgboost" for Extreme Gradient Boosting

### Using a single model

*Note*: This example uses the [Differentiated Thyroid Cancer Recurrence data from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence). Additionally,
if stratification is requested and one of the regularized models is used, then stratification will also be performed
on the training data used for `cv.glmnet`. In this case, the `foldid` parameter in `cv.glmnet` will be used to retain
the relative proportions in the target variable.

```R
# Set url for Thyroid Recurrence data from UCI Machine Learning Repository. This data has 383 instances and 16 features
url <- "https://archive.ics.uci.edu/static/public/915/differentiated+thyroid+cancer+recurrence.zip"

# Set file destination
dest_file <- file.path(getwd(), "thyroid.zip")

# Download zip file
download.file(url, dest_file)

# Unzip file
unzip(zipfile = dest_file, files = "Thyroid_Diff.csv")

thyroid_data <- read.csv("Thyroid_Diff.csv")

# Load the package
library(vswift)

# Model arguments; nfolds is the number of folds for `cv.glmnet`
map_args <- list(regularized_logistic = list(alpha = 1, nfolds = 3))

# Perform train-test split and cross-validation with stratified sampling
results <- classCV(
  data = thyroid_data,
  target = "Recurred",
  models = "regularized_logistic",
  model_params = list(map_args = map_args, rule = "1se", verbose = TRUE), # rule can be "min" or "1se"
  train_params = list(
    split = 0.8,
    n_folds = 5,
    standardize = TRUE,
    stratified = TRUE,
    random_seed = 50
  ),
  save = list(models = TRUE) # Saves both `cv.glmnet` and `glmnet` model
)

# Also valid, the target variable can refer to the column index
results <- classCV(
  data = thyroid_data,
  target = 17,
  models = "regularized_logistic",
  model_params = list(map_args = map_args, rule = "1se", verbose = TRUE),
  train_params = list(
    split = 0.8,
    n_folds = 5,
    standardize = TRUE,
    stratified = TRUE,
    random_seed = 50
  ),
  save = list(models = TRUE)
)

# Formula method can be used
results <- classCV(
  formula = Recurred ~ .,
  data = thyroid_data,
  models = "regularized_logistic",
  model_params = list(map_args = map_args, rule = "1se", verbose = TRUE),
  train_params = list(
    split = 0.8,
    n_folds = 5,
    standardize = TRUE,
    stratified = TRUE,
    random_seed = 50
  ),
  save = list(models = TRUE)
)

```

**Output Message**
```
Model: regularized_logistic | Partition: Train-Test Split | Optimal lambda: 0.06556 (nested 3-fold cross-validation using '1se' rule) 
Model: regularized_logistic | Partition: Fold 1 | Optimal lambda: 0.01357 (nested 3-fold cross-validation using '1se' rule) 
Model: regularized_logistic | Partition: Fold 2 | Optimal lambda: 0.04880 (nested 3-fold cross-validation using '1se' rule) 
Model: regularized_logistic | Partition: Fold 3 | Optimal lambda: 0.01226 (nested 3-fold cross-validation using '1se' rule) 
Model: regularized_logistic | Partition: Fold 4 | Optimal lambda: 0.06464 (nested 3-fold cross-validation using '1se' rule) 
Model: regularized_logistic | Partition: Fold 5 | Optimal lambda: 0.00847 (nested 3-fold cross-validation using '1se' rule)
```

Print optimal lambda values.
```R
print(results$metrics$regularized_logistic$optimal_lambdas)
```
**Output Message**
```
      split       fold1       fold2       fold3       fold4       fold5
0.065562223 0.013572555 0.048797016 0.012261040 0.064639632 0.008467439
```

`classCV` produces a vswift object which can be used for custom printing and plotting of performance metrics by using
the `print` and `plot` functions.

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

**Output**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Regularized Logistic Regression 

Formula: Recurred ~ .

Number of Features: 16

Classes: No, Yes

Training Parameters: list(split = 0.8, n_folds = 5, standardize = TRUE, stratified = TRUE, random_seed = 50, remove_obs = FALSE)

Model Parameters: list(map_args = list(regularized_logistic = list(alpha = 1, nfolds = 3)), rule = "1se", final_model = FALSE, verbose = TRUE)

Unlabeled Data: 0

Incomplete Labeled Data: 0

Sample Size (Complete Data): 383

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = NULL, future.seed = NULL)



 Training 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.95 

Class:   Precision:  Recall:       F1:

No             0.94     1.00      0.96 
Yes            0.99     0.83      0.90 


 Test 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.94 

Class:   Precision:  Recall:       F1:

No             0.93     0.98      0.96 
Yes            0.95     0.82      0.88 


 Cross-validation (CV) 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.95 ± 0.01 (SD) 

Class:       Average Precision:        Average Recall:            Average F1:

No             0.94 ± 0.01 (SD)       0.99 ± 0.01 (SD)       0.97 ± 0.01 (SD) 
Yes            0.98 ± 0.03 (SD)       0.84 ± 0.04 (SD)       0.91 ± 0.02 (SD) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot model evaluation metrics
plot(results, split = TRUE, cv = TRUE, path = getwd())
```

<details>
  
  <summary><strong>Plots</strong></summary>
  
  ![image](assets/thyroid/regularized_logistic_regression_cv_classification_accuracy.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_f1_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_f1_Yes.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_precision_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_precision_Yes.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_recall_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_cv_recall_Yes.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_classification_accuracy.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_f1_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_f1_Yes.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_precision_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_precision_Yes.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_recall_No.png)
  ![image](assets/thyroid/regularized_logistic_regression_train_test_recall_Yes.png)

</details>

### Producing ROC Curves with AUC scores
ROC curves are currently only available for binary classification tasks. The ROC curve plots the True Positive Rate (TPR)
against the False Positive Rate (FPR) at various classification thresholds. To generate the ROC curve, both the
models and preprocessed data must be saved.

```R

results <- classCV(
  formula = Recurred ~ .,
  data = thyroid_data,
  models = "naivebayes",
  train_params = list(
    split = 0.8,
    n_folds = 5,
    standardize = TRUE,
    stratified = TRUE,
    random_seed = 50
  ),
  save = list(models = TRUE, data = TRUE)
)

# Will derive thresholds from the probabilities
output <- roc_curve(results, return_output = TRUE, thresholds = NULL, path = getwd())

```

**Output**

```
Warning message:
In .create_dictionary(x$class_summary$classes, TRUE) :
  creating keys for target variable for `roc_curve`;
  classes are now encoded: No = 0, Yes = 1
```

![image](assets/roc/naivebayes_train_test_roc_curve.png)
![image](assets/roc/naivebayes_cv_roc_curve.png)

Output consists of a list containing thresholds used to generate the ROC, target labels, False Positive Rates (FPR),
True Positive Rates, Area Under The Curve (AUC), and Youdin's Index for all training and validation sets for each model.

```R

print(output)
```
<details>
    <summary><strong>Output</strong></summary>

    $naivebayes
    $naivebayes$split
    $naivebayes$split$train
    $naivebayes$split$train$thresholds
      [1] 0.000000e+00 3.292335e-11 7.567593e-10 9.671862e-10 9.674596e-10 9.920730e-10 1.068093e-09 1.085581e-09 1.095783e-09 1.165776e-09 1.299856e-09 1.513164e-09 1.961420e-09 2.119067e-09
     [15] 2.178872e-09 2.327707e-09 2.620570e-09 2.657457e-09 2.808008e-09 3.357334e-09 3.626539e-09 4.766705e-09 5.451639e-09 5.472565e-09 5.568416e-09 6.204016e-09 7.303997e-09 7.966935e-09
     [29] 1.267271e-08 1.429559e-08 2.263959e-08 3.664297e-08 3.838336e-08 4.020028e-08 4.107513e-08 4.541475e-08 5.677108e-08 6.512369e-08 1.236402e-07 1.648680e-07 2.739243e-07 4.464377e-07
     [43] 4.516064e-07 4.596298e-07 7.187438e-07 1.106685e-06 1.109952e-06 1.116412e-06 1.125376e-06 1.208703e-06 1.225408e-06 1.276823e-06 1.276944e-06 1.280352e-06 1.280714e-06 1.298511e-06
     [57] 1.299367e-06 1.313297e-06 1.317348e-06 1.332048e-06 1.333430e-06 1.354929e-06 1.384097e-06 1.453186e-06 1.492447e-06 1.495403e-06 1.543244e-06 1.597167e-06 1.795839e-06 2.076651e-06
     [71] 2.176292e-06 2.373536e-06 2.469398e-06 2.475990e-06 2.476690e-06 2.512762e-06 2.536911e-06 2.566283e-06 2.676615e-06 2.891861e-06 2.900606e-06 2.927204e-06 2.961094e-06 2.999331e-06
     [85] 3.003939e-06 3.081391e-06 3.247960e-06 3.320487e-06 3.362171e-06 3.420156e-06 3.460013e-06 3.589619e-06 3.674059e-06 3.720569e-06 4.015897e-06 4.396601e-06 5.228124e-06 5.416692e-06
     [99] 6.214757e-06 6.530947e-06 6.954918e-06 8.543021e-06 9.031037e-06 1.014178e-05 1.032015e-05 1.228709e-05 1.287021e-05 1.299238e-05 1.441210e-05 1.472310e-05 1.486118e-05 1.509146e-05
    [113] 1.586036e-05 1.595818e-05 1.606109e-05 1.626086e-05 1.669814e-05 1.682510e-05 1.770623e-05 1.773461e-05 1.995373e-05 2.070606e-05 2.235160e-05 2.713291e-05 2.720307e-05 3.036887e-05
    [127] 3.715620e-05 3.725978e-05 4.242795e-05 4.716603e-05 5.080903e-05 5.277676e-05 5.403061e-05 6.124046e-05 6.209479e-05 6.397941e-05 7.014515e-05 8.000685e-05 9.793564e-05 1.025965e-04
    [141] 1.052332e-04 1.072189e-04 1.200743e-04 1.225824e-04 1.340076e-04 1.548793e-04 1.914411e-04 2.053942e-04 2.193432e-04 2.836189e-04 2.943647e-04 2.983238e-04 3.711636e-04 5.014240e-04
    [155] 5.249065e-04 5.406552e-04 5.618462e-04 6.636290e-04 6.723283e-04 6.852598e-04 6.959785e-04 7.599133e-04 9.143529e-04 1.149699e-03 1.188414e-03 1.272195e-03 1.788190e-03 1.813092e-03
    [169] 1.983340e-03 2.033641e-03 2.191855e-03 2.740802e-03 2.843030e-03 3.722502e-03 3.798650e-03 3.865984e-03 4.658288e-03 6.388180e-03 6.972296e-03 8.171811e-03 8.367709e-03 1.233570e-02
    [183] 1.328297e-02 1.760014e-02 1.820398e-02 1.920872e-02 3.209828e-02 3.374577e-02 3.970853e-02 4.101716e-02 4.295397e-02 4.314777e-02 1.035270e-01 1.509014e-01 1.578062e-01 1.793192e-01
    [197] 2.565518e-01 2.648662e-01 2.666873e-01 2.965559e-01 2.978384e-01 3.234871e-01 3.334080e-01 3.418789e-01 3.488650e-01 3.999830e-01 5.551160e-01 5.703151e-01 5.704454e-01 5.985504e-01
    [211] 6.168398e-01 6.372939e-01 6.731577e-01 7.048432e-01 7.175881e-01 7.459657e-01 7.655506e-01 7.678166e-01 8.147317e-01 8.299756e-01 9.511289e-01 9.518912e-01 9.527830e-01 9.730970e-01
    [225] 9.781862e-01 9.907939e-01 9.956686e-01 9.987316e-01 9.990302e-01 9.992493e-01 9.994424e-01 9.994607e-01 9.996027e-01 9.996312e-01 9.997995e-01 9.998646e-01 9.998658e-01 9.999340e-01
    [239] 9.999403e-01 9.999436e-01 9.999539e-01 9.999613e-01 9.999727e-01 9.999778e-01 9.999793e-01 9.999825e-01 9.999835e-01 9.999836e-01 9.999860e-01 9.999888e-01 9.999917e-01 9.999930e-01
    [253] 9.999931e-01 9.999973e-01 9.999975e-01 9.999976e-01 9.999984e-01 9.999984e-01 9.999985e-01 9.999986e-01 9.999987e-01 9.999992e-01 9.999994e-01 9.999997e-01 9.999998e-01 9.999998e-01
    [267] 9.999999e-01 9.999999e-01 9.999999e-01 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    [281] 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$split$train$probs
      [1] 4.020028e-08 1.068093e-09 3.589619e-06 4.716603e-05 4.516064e-07 5.618462e-04 1.014178e-05 5.014240e-04 2.476690e-06 2.469398e-06 1.276823e-06 9.031037e-06 1.429559e-08 1.228709e-05
     [15] 1.543244e-06 1.495403e-06 1.276823e-06 3.003939e-06 5.406552e-04 2.927204e-06 5.472565e-09 3.036887e-05 4.015897e-06 1.597167e-06 1.298511e-06 1.983340e-03 1.384097e-06 1.313297e-06
     [29] 1.200743e-04 3.357334e-09 1.453186e-06 2.720307e-05 1.770623e-05 1.276944e-06 5.416692e-06 1.495403e-06 1.548793e-04 1.961420e-09 1.354929e-06 2.891861e-06 6.209479e-05 9.793564e-05
     [43] 2.900606e-06 5.228124e-06 4.658288e-03 3.003939e-06 5.568416e-09 1.486118e-05 2.999331e-06 1.280714e-06 2.900606e-06 3.362171e-06 1.626086e-05 1.773461e-05 1.995373e-05 5.277676e-05
     [57] 1.333430e-06 4.107513e-08 2.739243e-07 3.798650e-03 6.636290e-04 5.703151e-01 1.035270e-01 2.657457e-09 4.295397e-02 2.965559e-01 3.999830e-01 7.175881e-01 9.987316e-01 1.441210e-05
     [71] 6.852598e-04 1.188414e-03 6.124046e-05 5.677108e-08 3.374577e-02 6.959785e-04 9.781862e-01 9.994424e-01 9.999984e-01 9.999973e-01 9.999793e-01 2.620570e-09 9.999825e-01 3.488650e-01
     [85] 1.793192e-01 1.000000e+00 2.666873e-01 8.543021e-06 7.014515e-05 5.451639e-09 4.596298e-07 6.972296e-03 1.149699e-03 1.813092e-03 2.740802e-03 1.606109e-05 4.242795e-05 1.233570e-02
     [99] 7.966935e-09 1.052332e-04 1.299238e-05 1.788190e-03 2.943647e-04 1.682510e-05 2.033641e-03 6.723283e-04 1.328297e-02 9.143529e-04 3.209828e-02 9.671862e-10 4.314777e-02 8.147317e-01
    [113] 5.249065e-04 3.234871e-01 1.820398e-02 2.053942e-04 5.985504e-01 7.048432e-01 4.101716e-02 1.267271e-08 6.168398e-01 2.843030e-03 7.459657e-01 2.648662e-01 6.388180e-03 8.367709e-03
    [127] 7.678166e-01 1.509014e-01 1.095783e-09 3.334080e-01 9.996312e-01 9.999835e-01 9.998646e-01 9.994607e-01 9.997995e-01 9.999975e-01 7.303997e-09 9.907939e-01 9.999340e-01 9.999976e-01
    [141] 9.999888e-01 9.999860e-01 9.999999e-01 9.999985e-01 9.999999e-01 9.999994e-01 9.999931e-01 9.730970e-01 8.299756e-01 9.992493e-01 9.511289e-01 6.731577e-01 1.000000e+00 1.085581e-09
    [155] 9.999539e-01 1.000000e+00 9.999613e-01 2.978384e-01 9.998658e-01 9.999930e-01 6.372939e-01 9.999999e-01 9.920730e-10 1.000000e+00 1.000000e+00 1.000000e+00 1.920872e-02 9.999998e-01
    [169] 1.000000e+00 1.000000e+00 4.766705e-09 9.999997e-01 1.000000e+00 1.000000e+00 9.999998e-01 1.000000e+00 1.000000e+00 1.000000e+00 9.674596e-10 1.648680e-07 1.000000e+00 1.000000e+00
    [183] 1.000000e+00 1.000000e+00 1.000000e+00 6.214757e-06 3.865984e-03 1.109952e-06 3.722502e-03 2.327707e-09 1.578062e-01 2.808008e-09 2.076651e-06 3.247960e-06 1.068093e-09 3.838336e-08
    [197] 7.599133e-04 9.920730e-10 7.655506e-01 6.397941e-05 3.420156e-06 1.760014e-02 2.836189e-04 9.518912e-01 3.674059e-06 7.567593e-10 5.704454e-01 4.464377e-07 2.961094e-06 1.669814e-05
    [211] 2.676615e-06 1.280352e-06 1.025965e-04 5.551160e-01 1.340076e-04 2.070606e-05 1.276823e-06 2.565518e-01 3.970853e-02 1.332048e-06 1.165776e-09 1.595818e-05 2.119067e-09 1.586036e-05
    [225] 2.191855e-03 1.225824e-04 8.000685e-05 1.354929e-06 1.272195e-03 2.193432e-04 9.999778e-01 9.999436e-01 1.000000e+00 9.999727e-01 3.418789e-01 9.999987e-01 9.999992e-01 9.527830e-01
    [239] 9.999836e-01 3.292335e-11 9.999986e-01 1.000000e+00 9.999984e-01 1.000000e+00 8.171811e-03 9.999403e-01 1.000000e+00 2.178872e-09 1.000000e+00 1.000000e+00 9.956686e-01 2.263959e-08
    [253] 3.626539e-09 9.990302e-01 6.204016e-09 7.187438e-07 1.472310e-05 2.566283e-06 2.713291e-05 3.320487e-06 2.373536e-06 1.317348e-06 4.541475e-08 5.403061e-05 1.509146e-05 1.125376e-06
    [267] 4.396601e-06 1.072189e-04 6.512369e-08 3.674059e-06 1.225408e-06 2.235160e-05 1.106685e-06 1.116412e-06 2.536911e-06 1.116412e-06 2.176292e-06 3.664297e-08 3.720569e-06 3.715620e-05
    [281] 1.513164e-09 1.287021e-05 2.983238e-04 9.996027e-01 9.999917e-01 3.460013e-06 1.208703e-06 1.032015e-05 1.236402e-07 1.299856e-09 3.725978e-05 1.914411e-04 6.530947e-06 1.492447e-06
    [295] 3.003939e-06 5.080903e-05 1.276944e-06 6.954918e-06 3.081391e-06 2.475990e-06 3.711636e-04 1.795839e-06 1.299367e-06 1.332048e-06 2.512762e-06
    
    $naivebayes$split$train$labels
     No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
     No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No Yes Yes Yes Yes Yes Yes Yes  No Yes Yes Yes Yes Yes  No  No  No  No  No  No  No  No  No  No  No 
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   1   1   1   1   1   0   1   1   1   1   1   0   0   0   0   0   0   0   0   0   0   0 
     No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No Yes Yes  No Yes Yes Yes Yes Yes Yes Yes  No Yes Yes Yes Yes Yes Yes Yes Yes Yes Yes 
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   0   1   1   1   1   1   1   1   0   1   1   1   1   1   1   1   1   1   1 
    Yes Yes Yes Yes Yes Yes  No Yes Yes  No  No Yes Yes Yes Yes  No Yes Yes Yes  No Yes Yes Yes  No Yes Yes Yes Yes Yes Yes Yes  No  No Yes Yes Yes Yes Yes  No  No  No  No  No  No  No  No  No  No  No 
      1   1   1   1   1   1   0   1   1   0   0   1   1   1   1   0   1   1   1   0   1   1   1   0   1   1   1   1   1   1   1   0   0   1   1   1   1   1   0   0   0   0   0   0   0   0   0   0   0 
     No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No Yes Yes Yes Yes Yes Yes Yes Yes Yes  No Yes Yes Yes Yes Yes 
      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   1   1   1   1   1   1   1   0   1   1   1   1   1 
    Yes Yes  No Yes Yes Yes  No  No Yes  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No Yes Yes Yes  No  No  No  No  No  No  No  No  No 
      1   1   0   1   1   1   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   1   0   0   0   0   0   0   0   0   0 
     No  No  No  No  No  No  No  No  No  No  No 
      0   0   0   0   0   0   0   0   0   0   0 
    
    $naivebayes$split$train$metrics
    $naivebayes$split$train$metrics$fpr
      [1] 1.000000000 1.000000000 0.995475113 0.990950226 0.986425339 0.981900452 0.972850679 0.963800905 0.959276018 0.954751131 0.950226244 0.945701357 0.941176471 0.936651584 0.932126697 0.927601810
     [17] 0.923076923 0.918552036 0.914027149 0.909502262 0.904977376 0.900452489 0.895927602 0.891402715 0.886877828 0.882352941 0.877828054 0.873303167 0.868778281 0.864253394 0.859728507 0.855203620
     [33] 0.850678733 0.846153846 0.841628959 0.837104072 0.832579186 0.828054299 0.823529412 0.819004525 0.814479638 0.809954751 0.805429864 0.800904977 0.796380090 0.791855204 0.787330317 0.782805430
     [49] 0.773755656 0.769230769 0.764705882 0.760180995 0.746606335 0.737556561 0.733031674 0.728506787 0.723981900 0.719457014 0.714932127 0.710407240 0.701357466 0.696832579 0.687782805 0.683257919
     [65] 0.678733032 0.674208145 0.665158371 0.660633484 0.656108597 0.651583710 0.647058824 0.642533937 0.638009050 0.633484163 0.628959276 0.624434389 0.619909502 0.615384615 0.610859729 0.606334842
     [81] 0.601809955 0.592760181 0.588235294 0.583710407 0.579185520 0.565610860 0.561085973 0.556561086 0.552036199 0.547511312 0.542986425 0.538461538 0.533936652 0.524886878 0.520361991 0.515837104
     [97] 0.511312217 0.506787330 0.502262443 0.497737557 0.493212670 0.488687783 0.484162896 0.479638009 0.475113122 0.470588235 0.466063348 0.461538462 0.457013575 0.452488688 0.447963801 0.443438914
    [113] 0.438914027 0.434389140 0.429864253 0.425339367 0.420814480 0.416289593 0.411764706 0.407239819 0.402714932 0.398190045 0.393665158 0.389140271 0.384615385 0.380090498 0.375565611 0.371040724
    [129] 0.366515837 0.361990950 0.357466063 0.352941176 0.348416290 0.343891403 0.339366516 0.334841629 0.330316742 0.325791855 0.321266968 0.316742081 0.312217195 0.307692308 0.303167421 0.298642534
    [145] 0.294117647 0.289592760 0.285067873 0.280542986 0.276018100 0.271493213 0.266968326 0.262443439 0.262443439 0.257918552 0.253393665 0.248868778 0.244343891 0.239819005 0.235294118 0.230769231
    [161] 0.226244344 0.226244344 0.221719457 0.217194570 0.212669683 0.208144796 0.203619910 0.199095023 0.194570136 0.190045249 0.185520362 0.180995475 0.176470588 0.171945701 0.167420814 0.162895928
    [177] 0.158371041 0.153846154 0.149321267 0.144796380 0.144796380 0.140271493 0.135746606 0.131221719 0.126696833 0.122171946 0.117647059 0.113122172 0.113122172 0.108597285 0.104072398 0.099547511
    [193] 0.095022624 0.090497738 0.090497738 0.085972851 0.085972851 0.081447964 0.076923077 0.076923077 0.072398190 0.067873303 0.063348416 0.063348416 0.063348416 0.063348416 0.058823529 0.054298643
    [209] 0.049773756 0.045248869 0.040723982 0.036199095 0.036199095 0.036199095 0.031674208 0.027149321 0.022624434 0.018099548 0.018099548 0.013574661 0.013574661 0.013574661 0.009049774 0.009049774
    [225] 0.009049774 0.009049774 0.009049774 0.009049774 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887 0.004524887
    [241] 0.004524887 0.004524887 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000
    [257] 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000
    [273] 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000
    [289] 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000
    
    $naivebayes$split$train$metrics$tpr
      [1] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
     [18] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
     [35] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
     [52] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
     [69] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
     [86] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [103] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [120] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [137] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 0.98809524
    [154] 0.98809524 0.98809524 0.98809524 0.98809524 0.98809524 0.98809524 0.98809524 0.98809524 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048
    [171] 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.97619048 0.96428571 0.96428571 0.96428571 0.96428571 0.96428571 0.96428571 0.96428571
    [188] 0.96428571 0.95238095 0.95238095 0.95238095 0.95238095 0.95238095 0.95238095 0.94047619 0.94047619 0.92857143 0.92857143 0.92857143 0.91666667 0.91666667 0.91666667 0.91666667 0.90476190
    [205] 0.89285714 0.88095238 0.88095238 0.88095238 0.88095238 0.88095238 0.88095238 0.88095238 0.86904762 0.85714286 0.85714286 0.85714286 0.85714286 0.85714286 0.84523810 0.84523810 0.83333333
    [222] 0.82142857 0.82142857 0.80952381 0.79761905 0.78571429 0.77380952 0.76190476 0.76190476 0.75000000 0.73809524 0.72619048 0.71428571 0.70238095 0.69047619 0.67857143 0.66666667 0.65476190
    [239] 0.64285714 0.63095238 0.61904762 0.60714286 0.60714286 0.59523810 0.58333333 0.57142857 0.55952381 0.54761905 0.53571429 0.52380952 0.51190476 0.50000000 0.48809524 0.47619048 0.46428571
    [256] 0.45238095 0.44047619 0.42857143 0.41666667 0.40476190 0.39285714 0.38095238 0.36904762 0.35714286 0.34523810 0.33333333 0.32142857 0.30952381 0.29761905 0.28571429 0.27380952 0.26190476
    [273] 0.25000000 0.23809524 0.22619048 0.21428571 0.20238095 0.19047619 0.17857143 0.16666667 0.15476190 0.14285714 0.13095238 0.11904762 0.10714286 0.09523810 0.08333333 0.07142857 0.05952381
    [290] 0.04761905 0.03571429 0.02380952 0.01190476
    
    
    $naivebayes$split$train$auc
    [1] 0.9818466
    
    $naivebayes$split$train$youdins_indx
    [1] 0.1509014
    
    
    $naivebayes$split$test
    $naivebayes$split$test$thresholds
     [1] 0.000000e+00 3.102838e-11 5.637034e-11 1.655982e-09 1.945901e-09 2.305786e-09 2.415472e-09 2.360352e-08 5.890455e-08 9.477141e-08 4.327154e-07 1.125376e-06 1.288167e-06 1.299367e-06
    [15] 1.314411e-06 1.450587e-06 1.453186e-06 2.192172e-06 2.273632e-06 2.491103e-06 2.884367e-06 3.003939e-06 3.118082e-06 3.210786e-06 3.362171e-06 3.724963e-06 4.908487e-06 9.004547e-06
    [29] 9.520265e-06 1.310164e-05 1.384745e-05 1.470508e-05 2.248991e-05 2.253779e-05 3.896095e-05 5.420653e-05 6.063447e-05 9.602061e-05 1.039374e-04 1.492905e-04 3.989868e-04 5.603026e-04
    [43] 7.207525e-04 1.258081e-02 1.371243e-02 3.231545e-02 3.831589e-02 4.434028e-02 7.110187e-02 7.521754e-02 2.694066e-01 6.344531e-01 7.081465e-01 8.980074e-01 9.014546e-01 9.515796e-01
    [57] 9.687055e-01 9.696865e-01 9.905595e-01 9.932429e-01 9.961164e-01 9.979787e-01 9.982557e-01 9.985101e-01 9.986610e-01 9.999740e-01 9.999876e-01 9.999929e-01 9.999982e-01 9.999999e-01
    [71] 9.999999e-01 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$split$test$probs
     [1] 5.420653e-05 9.696865e-01 1.371243e-02 9.999999e-01 9.602061e-05 3.362171e-06 9.999999e-01 9.004547e-06 9.999876e-01 7.110187e-02 1.314411e-06 4.908487e-06 1.288167e-06 6.063447e-05
    [15] 2.491103e-06 2.694066e-01 3.118082e-06 1.000000e+00 9.014546e-01 2.415472e-09 9.999982e-01 3.102838e-11 9.932429e-01 9.985101e-01 9.520265e-06 9.905595e-01 5.637034e-11 2.360352e-08
    [29] 2.884367e-06 2.305786e-09 9.687055e-01 4.327154e-07 7.081465e-01 3.896095e-05 1.258081e-02 9.986610e-01 1.453186e-06 1.000000e+00 9.982557e-01 9.515796e-01 7.521754e-02 1.655982e-09
    [43] 9.979787e-01 1.310164e-05 1.000000e+00 1.000000e+00 1.450587e-06 5.603026e-04 3.989868e-04 2.273632e-06 1.384745e-05 9.999740e-01 1.945901e-09 1.000000e+00 1.039374e-04 8.980074e-01
    [57] 4.434028e-02 2.192172e-06 9.477141e-08 9.999929e-01 1.125376e-06 1.470508e-05 6.344531e-01 3.831589e-02 2.253779e-05 1.000000e+00 1.655982e-09 1.299367e-06 9.961164e-01 1.000000e+00
    [71] 1.492905e-04 7.207525e-04 3.231545e-02 3.210786e-06 5.890455e-08 3.003939e-06 2.248991e-05 3.724963e-06
    
    $naivebayes$split$test$labels
     No Yes  No Yes  No  No Yes  No Yes Yes  No  No  No  No  No  No  No Yes  No  No Yes  No  No Yes  No Yes  No  No  No  No Yes  No  No  No  No Yes  No Yes Yes Yes  No  No  No  No Yes Yes  No  No  No 
      0   1   0   1   0   0   1   0   1   1   0   0   0   0   0   0   0   1   0   0   1   0   0   1   0   1   0   0   0   0   1   0   0   0   0   1   0   1   1   1   0   0   0   0   1   1   0   0   0 
     No  No Yes  No Yes  No Yes  No  No  No Yes  No  No Yes  No  No Yes  No  No Yes Yes  No  No  No  No  No  No  No  No 
      0   0   1   0   1   0   1   0   0   0   1   0   0   1   0   0   1   0   0   1   1   0   0   0   0   0   0   0   0 
    
    $naivebayes$split$test$metrics
    $naivebayes$split$test$metrics$fpr
     [1] 1.00000000 1.00000000 0.98148148 0.96296296 0.92592593 0.90740741 0.88888889 0.87037037 0.85185185 0.83333333 0.81481481 0.79629630 0.77777778 0.75925926 0.74074074 0.72222222 0.70370370
    [18] 0.68518519 0.66666667 0.64814815 0.62962963 0.61111111 0.59259259 0.57407407 0.55555556 0.53703704 0.51851852 0.50000000 0.48148148 0.46296296 0.44444444 0.42592593 0.40740741 0.38888889
    [35] 0.37037037 0.35185185 0.33333333 0.31481481 0.29629630 0.27777778 0.25925926 0.24074074 0.22222222 0.20370370 0.18518519 0.16666667 0.14814815 0.12962963 0.11111111 0.11111111 0.09259259
    [52] 0.07407407 0.07407407 0.05555556 0.05555556 0.03703704 0.03703704 0.03703704 0.03703704 0.03703704 0.01851852 0.01851852 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$split$test$metrics$tpr
     [1] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [18] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [35] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 0.95833333 0.95833333
    [52] 0.95833333 0.91666667 0.91666667 0.87500000 0.87500000 0.83333333 0.79166667 0.75000000 0.70833333 0.70833333 0.66666667 0.66666667 0.62500000 0.58333333 0.54166667 0.50000000 0.45833333
    [69] 0.41666667 0.37500000 0.33333333 0.29166667 0.25000000 0.20833333 0.16666667 0.12500000 0.08333333 0.04166667
    
    
    $naivebayes$split$test$auc
    [1] 0.9772377
    
    $naivebayes$split$test$youdins_indx
    [1] 0.07110187
    
    
    
    $naivebayes$cv
    $naivebayes$cv$fold1
    $naivebayes$cv$fold1$thresholds
     [1] 0.000000e+00 1.179031e-11 2.126408e-11 2.894429e-10 3.646142e-10 3.881288e-10 4.310325e-10 5.545877e-10 6.836670e-10 8.173240e-10 1.467840e-09 1.893783e-09 2.295993e-09 3.779048e-09
    [15] 1.912345e-07 2.697474e-07 3.186805e-07 3.601236e-07 3.872300e-07 3.909683e-07 3.968867e-07 4.130678e-07 4.355029e-07 5.083762e-07 5.327279e-07 5.794200e-07 5.805344e-07 5.945570e-07
    [29] 7.907472e-07 7.952150e-07 9.925731e-07 2.875101e-06 3.329599e-06 4.668524e-06 5.065577e-06 5.201129e-06 6.912615e-06 2.752126e-05 2.971633e-05 3.954587e-05 7.125547e-05 8.078141e-05
    [43] 9.743859e-05 1.104944e-04 2.092174e-04 1.920150e-03 4.118369e-03 4.886238e-03 2.498849e-02 6.853987e-02 7.370398e-02 2.187185e-01 3.056246e-01 3.907904e-01 5.696868e-01 5.890897e-01
    [57] 7.825238e-01 8.252651e-01 8.792337e-01 9.041457e-01 9.042361e-01 9.849095e-01 9.875460e-01 9.894312e-01 9.982790e-01 9.998389e-01 9.999236e-01 9.999463e-01 9.999683e-01 9.999920e-01
    [71] 9.999958e-01 9.999999e-01 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$cv$fold1$probs
     [1] 2.126408e-11 2.092174e-04 4.130678e-07 2.894429e-10 6.836670e-10 9.998389e-01 4.355029e-07 1.179031e-11 7.952150e-07 1.920150e-03 5.794200e-07 5.201129e-06 9.999958e-01 1.467840e-09
    [15] 7.370398e-02 3.646142e-10 3.881288e-10 3.909683e-07 8.078141e-05 3.056246e-01 9.982790e-01 6.912615e-06 9.999683e-01 9.875460e-01 3.329599e-06 3.186805e-07 9.999463e-01 2.187185e-01
    [29] 1.000000e+00 1.912345e-07 7.907472e-07 3.907904e-01 1.893783e-09 9.925731e-07 3.968867e-07 2.752126e-05 3.779048e-09 4.118369e-03 9.743859e-05 1.000000e+00 9.894312e-01 9.999920e-01
    [43] 7.825238e-01 1.000000e+00 2.498849e-02 4.668524e-06 4.886238e-03 5.945570e-07 1.000000e+00 3.601236e-07 7.125547e-05 3.954587e-05 8.173240e-10 9.041457e-01 5.065577e-06 6.853987e-02
    [57] 2.971633e-05 4.310325e-10 1.104944e-04 2.697474e-07 8.252651e-01 9.042361e-01 5.083762e-07 5.545877e-10 8.792337e-01 5.327279e-07 3.872300e-07 5.805344e-07 2.295993e-09 9.999236e-01
    [71] 9.849095e-01 5.890897e-01 2.875101e-06 9.999999e-01 4.310325e-10 1.000000e+00 5.696868e-01
    
    $naivebayes$cv$fold1$labels
     No  No  No  No  No Yes  No  No  No  No  No  No Yes  No Yes  No  No  No  No  No Yes  No Yes Yes  No  No Yes Yes Yes  No  No  No  No  No  No  No  No  No  No Yes  No Yes Yes Yes  No  No  No  No Yes 
      0   0   0   0   0   1   0   0   0   0   0   0   1   0   1   0   0   0   0   0   1   0   1   1   0   0   1   1   1   0   0   0   0   0   0   0   0   0   0   1   0   1   1   1   0   0   0   0   1 
     No  No  No  No  No  No  No  No  No  No  No Yes  No  No  No Yes  No  No  No  No Yes Yes  No  No Yes  No Yes Yes 
      0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   1   0   0   0   0   1   1   0   0   1   0   1   1 
    
    $naivebayes$cv$fold1$metrics
    $naivebayes$cv$fold1$metrics$fpr
     [1] 1.00000000 1.00000000 0.98214286 0.96428571 0.94642857 0.92857143 0.91071429 0.87500000 0.85714286 0.83928571 0.82142857 0.80357143 0.78571429 0.76785714 0.75000000 0.73214286 0.71428571
    [18] 0.69642857 0.67857143 0.66071429 0.64285714 0.62500000 0.60714286 0.58928571 0.57142857 0.55357143 0.53571429 0.51785714 0.50000000 0.48214286 0.46428571 0.44642857 0.42857143 0.41071429
    [35] 0.39285714 0.37500000 0.35714286 0.33928571 0.32142857 0.30357143 0.28571429 0.26785714 0.25000000 0.23214286 0.21428571 0.19642857 0.17857143 0.16071429 0.14285714 0.12500000 0.10714286
    [52] 0.10714286 0.10714286 0.08928571 0.07142857 0.07142857 0.05357143 0.05357143 0.05357143 0.05357143 0.03571429 0.01785714 0.01785714 0.01785714 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$cv$fold1$metrics$tpr
     [1] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [18] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [35] 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000 1.00000000
    [52] 0.95238095 0.90476190 0.90476190 0.90476190 0.85714286 0.85714286 0.80952381 0.76190476 0.71428571 0.71428571 0.71428571 0.66666667 0.61904762 0.61904762 0.57142857 0.52380952 0.47619048
    [69] 0.42857143 0.38095238 0.33333333 0.28571429 0.23809524 0.19047619 0.14285714 0.09523810 0.04761905
    
    
    $naivebayes$cv$fold1$auc
    [1] 0.9719388
    
    $naivebayes$cv$fold1$youdins_indx
    [1] 0.07370398
    
    
    $naivebayes$cv$fold2
    $naivebayes$cv$fold2$thresholds
     [1] 0.000000e+00 1.046832e-09 2.583282e-09 2.653754e-09 3.208205e-09 8.009631e-09 3.772289e-08 4.392033e-07 1.076896e-06 1.548111e-06 1.572880e-06 1.607413e-06 1.648053e-06 1.652619e-06
    [15] 1.685809e-06 1.784179e-06 1.844469e-06 2.190796e-06 2.473192e-06 3.109782e-06 4.007786e-06 4.605397e-06 4.608395e-06 6.706597e-06 1.177476e-05 1.221843e-05 1.312520e-05 1.388134e-05
    [29] 2.523769e-05 4.395600e-05 4.556385e-05 5.115949e-05 5.653881e-05 7.012771e-05 8.118976e-05 8.230404e-05 1.479174e-04 1.480052e-04 3.992151e-04 4.385190e-04 5.025453e-04 6.039969e-04
    [43] 1.141787e-03 1.438733e-03 1.951549e-03 3.014924e-03 3.006687e-02 3.070906e-02 8.914674e-02 3.544441e-01 5.114110e-01 5.423707e-01 5.805072e-01 7.075259e-01 7.787423e-01 9.603903e-01
    [57] 9.792807e-01 9.930675e-01 9.993695e-01 9.997331e-01 9.997838e-01 9.998106e-01 9.999138e-01 9.999765e-01 9.999886e-01 9.999986e-01 9.999998e-01 9.999999e-01 9.999999e-01 1.000000e+00
    [71] 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$cv$fold2$probs
     [1] 1.000000e+00 9.999886e-01 9.993695e-01 5.115949e-05 6.039969e-04 9.997838e-01 2.583282e-09 9.999138e-01 3.208205e-09 7.012771e-05 5.114110e-01 1.784179e-06 4.605397e-06 1.000000e+00
    [15] 9.930675e-01 5.805072e-01 1.177476e-05 8.118976e-05 3.992151e-04 8.009631e-09 4.608395e-06 6.706597e-06 9.998106e-01 1.480052e-04 4.395600e-05 1.000000e+00 1.000000e+00 1.438733e-03
    [29] 4.385190e-04 8.230404e-05 1.572880e-06 1.479174e-04 9.999765e-01 8.914674e-02 3.006687e-02 1.951549e-03 4.392033e-07 1.844469e-06 2.473192e-06 7.075259e-01 1.221843e-05 3.014924e-03
    [43] 2.523769e-05 9.999986e-01 9.999998e-01 1.046832e-09 1.685809e-06 7.787423e-01 1.141787e-03 9.792807e-01 4.007786e-06 1.000000e+00 5.423707e-01 9.999999e-01 9.999999e-01 2.653754e-09
    [57] 3.544441e-01 1.000000e+00 1.607413e-06 9.997331e-01 1.312520e-05 1.388134e-05 5.025453e-04 3.772289e-08 1.000000e+00 1.000000e+00 2.190796e-06 9.603903e-01 3.070906e-02 1.076896e-06
    [71] 1.000000e+00 1.648053e-06 5.653881e-05 1.548111e-06 3.109782e-06 4.556385e-05 1.652619e-06
    
    $naivebayes$cv$fold2$labels
    Yes Yes Yes  No  No Yes  No Yes  No  No  No  No  No Yes  No  No  No  No  No  No  No  No Yes  No  No Yes Yes  No  No  No  No  No Yes  No  No  No  No  No  No  No  No  No  No Yes Yes  No  No Yes  No 
      1   1   1   0   0   1   0   1   0   0   0   0   0   1   0   0   0   0   0   0   0   0   1   0   0   1   1   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   1   1   0   0   1   0 
    Yes  No Yes  No Yes Yes  No Yes Yes  No Yes  No  No Yes  No Yes Yes  No Yes  No  No Yes  No  No  No  No  No  No 
      1   0   1   0   1   1   0   1   1   0   1   0   0   1   0   1   1   0   1   0   0   1   0   0   0   0   0   0 
    
    $naivebayes$cv$fold2$metrics
    $naivebayes$cv$fold2$metrics$fpr
     [1] 1.00000000 1.00000000 0.98076923 0.96153846 0.94230769 0.92307692 0.90384615 0.88461538 0.86538462 0.84615385 0.82692308 0.80769231 0.78846154 0.76923077 0.75000000 0.73076923 0.71153846
    [18] 0.69230769 0.67307692 0.65384615 0.63461538 0.61538462 0.59615385 0.57692308 0.55769231 0.53846154 0.51923077 0.50000000 0.48076923 0.46153846 0.44230769 0.42307692 0.40384615 0.38461538
    [35] 0.36538462 0.34615385 0.32692308 0.30769231 0.28846154 0.26923077 0.25000000 0.25000000 0.23076923 0.21153846 0.19230769 0.17307692 0.15384615 0.13461538 0.11538462 0.09615385 0.09615385
    [52] 0.07692308 0.05769231 0.03846154 0.01923077 0.01923077 0.01923077 0.01923077 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$cv$fold2$metrics$tpr
     [1] 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    [39] 1.00 1.00 1.00 0.96 0.96 0.96 0.96 0.96 0.96 0.96 0.96 0.96 0.92 0.92 0.92 0.92 0.92 0.88 0.84 0.80 0.80 0.76 0.72 0.68 0.64 0.60 0.56 0.52 0.48 0.44 0.40 0.36 0.32 0.28 0.24 0.20 0.16 0.12
    [77] 0.08 0.04
    
    
    $naivebayes$cv$fold2$auc
    [1] 0.9765385
    
    $naivebayes$cv$fold2$youdins_indx
    [1] 0.7787423
    
    
    $naivebayes$cv$fold3
    $naivebayes$cv$fold3$thresholds
     [1] 0.000000e+00 2.244416e-13 2.527968e-13 1.031051e-11 1.084985e-11 1.085676e-11 1.454814e-11 1.524824e-11 1.632375e-11 2.344799e-11 5.958780e-11 1.005366e-10 5.275440e-10 4.287117e-08
    [15] 9.528412e-08 1.146614e-07 1.171095e-07 1.184564e-07 1.275471e-07 1.482298e-07 1.838814e-07 1.881519e-07 2.160163e-07 2.280952e-07 2.325834e-07 2.363990e-07 2.414463e-07 2.477593e-07
    [29] 2.832779e-07 5.023872e-07 5.642060e-07 7.476508e-07 8.041796e-07 1.305308e-06 1.509173e-06 1.600270e-06 1.812059e-06 2.144813e-06 2.650864e-06 4.756927e-06 5.736440e-06 5.776259e-06
    [43] 3.064135e-05 5.053945e-05 6.110384e-05 8.694901e-05 3.223210e-04 4.855596e-04 5.515783e-04 1.849525e-03 3.146537e-03 4.777321e-03 6.184621e-03 8.364389e-03 1.095744e-02 2.157026e-02
    [57] 2.191528e-02 5.055132e-02 9.513253e-02 1.640733e-01 5.797801e-01 7.426596e-01 8.750083e-01 9.525163e-01 9.612784e-01 9.695740e-01 9.977899e-01 9.999876e-01 9.999980e-01 9.999985e-01
    [71] 9.999999e-01 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$cv$fold3$probs
     [1] 5.736440e-06 1.000000e+00 9.999876e-01 4.777321e-03 7.426596e-01 9.999985e-01 1.000000e+00 5.515783e-04 5.055132e-02 1.000000e+00 1.305308e-06 1.171095e-07 4.756927e-06 2.527968e-13
    [15] 5.958780e-11 5.642060e-07 6.184621e-03 2.160163e-07 2.344799e-11 5.023872e-07 1.632375e-11 1.275471e-07 9.999980e-01 5.776259e-06 2.363990e-07 1.085676e-11 9.695740e-01 8.041796e-07
    [29] 2.832779e-07 9.977899e-01 1.184564e-07 8.364389e-03 9.999999e-01 1.812059e-06 2.414463e-07 9.513253e-02 1.849525e-03 1.600270e-06 1.095744e-02 1.881519e-07 1.000000e+00 2.191528e-02
    [43] 2.477593e-07 1.084985e-11 1.524824e-11 3.223210e-04 2.650864e-06 1.454814e-11 1.005366e-10 1.000000e+00 4.287117e-08 2.244416e-13 9.612784e-01 5.053945e-05 8.750083e-01 9.528412e-08
    [57] 2.325834e-07 1.509173e-06 1.482298e-07 1.838814e-07 5.275440e-10 1.640733e-01 2.157026e-02 9.525163e-01 5.797801e-01 1.031051e-11 4.855596e-04 1.146614e-07 3.146537e-03 2.280952e-07
    [71] 7.476508e-07 3.064135e-05 2.144813e-06 6.110384e-05 1.881519e-07 1.000000e+00 8.694901e-05
    
    $naivebayes$cv$fold3$labels
     No Yes Yes  No  No Yes Yes  No  No Yes  No  No  No  No  No  No  No  No  No  No  No  No Yes  No  No  No Yes  No  No Yes  No  No Yes  No  No  No  No  No  No  No Yes  No  No  No  No  No  No  No  No 
      0   1   1   0   0   1   1   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   1   0   0   1   0   0   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0 
    Yes  No  No Yes  No Yes  No  No  No  No  No  No Yes  No Yes  No  No  No  No  No  No  No  No  No  No  No Yes  No 
      1   0   0   1   0   1   0   0   0   0   0   0   1   0   1   0   0   0   0   0   0   0   0   0   0   0   1   0 
    
    $naivebayes$cv$fold3$metrics
    $naivebayes$cv$fold3$metrics$fpr
     [1] 1.00000000 1.00000000 0.98360656 0.96721311 0.95081967 0.93442623 0.91803279 0.90163934 0.88524590 0.86885246 0.85245902 0.83606557 0.81967213 0.80327869 0.78688525 0.77049180 0.75409836
    [18] 0.73770492 0.72131148 0.70491803 0.68852459 0.67213115 0.63934426 0.62295082 0.60655738 0.59016393 0.57377049 0.55737705 0.54098361 0.52459016 0.50819672 0.49180328 0.47540984 0.45901639
    [35] 0.44262295 0.42622951 0.40983607 0.39344262 0.37704918 0.36065574 0.34426230 0.32786885 0.31147541 0.29508197 0.27868852 0.26229508 0.24590164 0.22950820 0.21311475 0.19672131 0.18032787
    [52] 0.16393443 0.14754098 0.13114754 0.11475410 0.09836066 0.08196721 0.06557377 0.04918033 0.03278689 0.03278689 0.01639344 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$cv$fold3$metrics$tpr
     [1] 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
    [28] 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
    [55] 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.9375 0.9375 0.9375 0.8750 0.8125 0.7500 0.6875 0.6250 0.5625 0.5000 0.4375 0.3750 0.3125 0.2500 0.1875 0.1250 0.0625
    
    
    $naivebayes$cv$fold3$auc
    [1] 0.9907787
    
    $naivebayes$cv$fold3$youdins_indx
    [1] 0.1640733
    
    
    $naivebayes$cv$fold4
    $naivebayes$cv$fold4$thresholds
     [1] 0.000000e+00 7.091137e-10 1.265502e-09 4.189248e-09 5.330722e-09 6.031948e-09 7.922453e-09 6.510034e-08 6.935341e-08 8.738086e-08 1.007189e-06 1.787564e-06 1.796702e-06 1.860779e-06
    [15] 1.919512e-06 2.008677e-06 2.359348e-06 3.257911e-06 3.972564e-06 4.207580e-06 4.220332e-06 4.373681e-06 6.048511e-06 6.344941e-06 6.878250e-06 2.684353e-05 3.076176e-05 4.653905e-05
    [29] 4.789796e-05 6.823802e-05 7.804975e-05 1.021126e-04 1.215974e-04 2.183328e-04 2.737319e-04 2.888739e-04 7.486067e-04 1.023643e-03 1.656951e-03 2.126451e-03 2.181155e-03 3.999541e-03
    [43] 2.424190e-02 3.806705e-02 6.667543e-02 1.041541e-01 1.370152e-01 1.371574e-01 3.465967e-01 5.459612e-01 5.731090e-01 7.037640e-01 9.044430e-01 9.529544e-01 9.850998e-01 9.969255e-01
    [57] 9.970866e-01 9.999216e-01 9.999343e-01 9.999620e-01 9.999771e-01 9.999825e-01 9.999884e-01 9.999887e-01 9.999919e-01 9.999994e-01 9.999998e-01 9.999999e-01 1.000000e+00 1.000000e+00
    [71] 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$cv$fold4$probs
     [1] 6.510034e-08 9.850998e-01 4.789796e-05 2.183328e-04 1.656951e-03 4.653905e-05 2.424190e-02 1.265502e-09 4.207580e-06 1.860779e-06 4.373681e-06 9.999825e-01 2.126451e-03 1.860779e-06
    [15] 3.972564e-06 6.823802e-05 1.000000e+00 1.215974e-04 1.023643e-03 1.370152e-01 9.999216e-01 1.000000e+00 4.220332e-06 9.999884e-01 9.999998e-01 9.999887e-01 1.000000e+00 9.529544e-01
    [29] 5.330722e-09 7.922453e-09 2.888739e-04 1.007189e-06 5.731090e-01 3.999541e-03 4.189248e-09 3.076176e-05 1.796702e-06 7.037640e-01 2.684353e-05 9.969255e-01 1.000000e+00 6.935341e-08
    [43] 2.181155e-03 6.667543e-02 5.459612e-01 1.371574e-01 7.091137e-10 1.021126e-04 9.999994e-01 9.999771e-01 1.041541e-01 1.919512e-06 3.465967e-01 3.257911e-06 7.804975e-05 7.486067e-04
    [57] 9.999620e-01 1.787564e-06 6.048511e-06 6.344941e-06 3.806705e-02 6.878250e-06 9.044430e-01 2.737319e-04 9.999919e-01 4.220332e-06 1.000000e+00 2.359348e-06 9.999343e-01 2.008677e-06
    [71] 1.000000e+00 6.031948e-09 9.999999e-01 9.970866e-01 8.738086e-08 6.344941e-06
    
    $naivebayes$cv$fold4$labels
     No Yes  No  No  No  No  No  No  No  No  No Yes  No  No  No  No Yes  No  No  No Yes Yes  No Yes Yes Yes Yes Yes  No  No  No  No  No  No  No  No  No  No  No Yes Yes  No  No  No Yes  No  No  No Yes 
      0   1   0   0   0   0   0   0   0   0   0   1   0   0   0   0   1   0   0   0   1   1   0   1   1   1   1   1   0   0   0   0   0   0   0   0   0   0   0   1   1   0   0   0   1   0   0   0   1 
    Yes Yes  No Yes  No  No  No Yes  No  No  No  No  No Yes  No Yes  No Yes  No Yes  No Yes  No Yes Yes  No  No 
      1   1   0   1   0   0   0   1   0   0   0   0   0   1   0   1   0   1   0   1   0   1   0   1   1   0   0 
    
    $naivebayes$cv$fold4$metrics
    $naivebayes$cv$fold4$metrics$fpr
     [1] 1.00000000 1.00000000 0.98039216 0.96078431 0.94117647 0.92156863 0.90196078 0.88235294 0.86274510 0.84313725 0.82352941 0.80392157 0.78431373 0.76470588 0.72549020 0.70588235 0.68627451
    [18] 0.66666667 0.64705882 0.62745098 0.60784314 0.56862745 0.54901961 0.52941176 0.49019608 0.47058824 0.45098039 0.43137255 0.41176471 0.39215686 0.37254902 0.35294118 0.33333333 0.31372549
    [35] 0.29411765 0.27450980 0.25490196 0.23529412 0.21568627 0.19607843 0.17647059 0.15686275 0.13725490 0.11764706 0.09803922 0.07843137 0.07843137 0.05882353 0.03921569 0.03921569 0.03921569
    [52] 0.01960784 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$cv$fold4$metrics$tpr
     [1] 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    [39] 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.96 0.96 0.96 0.92 0.88 0.88 0.88 0.84 0.80 0.76 0.72 0.68 0.64 0.60 0.56 0.52 0.48 0.44 0.40 0.36 0.32 0.28 0.24 0.20 0.16 0.12 0.08 0.04
    
    
    $naivebayes$cv$fold4$auc
    [1] 0.9854902
    
    $naivebayes$cv$fold4$youdins_indx
    [1] 0.1041541
    
    
    $naivebayes$cv$fold5
    $naivebayes$cv$fold5$thresholds
     [1] 0.000000e+00 1.117280e-09 1.501349e-09 3.473973e-09 3.688143e-09 6.579601e-09 8.355749e-09 7.742514e-08 1.655608e-06 1.664325e-06 1.799282e-06 1.827851e-06 1.920745e-06 1.944447e-06
    [15] 2.040910e-06 3.168290e-06 3.686265e-06 4.239387e-06 4.379488e-06 4.539093e-06 4.760809e-06 5.009503e-06 5.069363e-06 5.128878e-06 5.871301e-06 7.136240e-06 9.823116e-06 1.006547e-05
    [29] 1.572257e-05 2.305073e-05 2.322587e-05 2.620194e-05 2.646258e-05 3.506807e-05 7.328366e-05 9.350814e-05 1.026141e-04 1.115303e-04 1.451176e-04 5.640481e-04 1.041805e-03 1.065068e-03
    [43] 1.722777e-03 1.983830e-03 2.431317e-03 4.605713e-03 7.490103e-03 7.750688e-03 1.191726e-02 2.315508e-02 4.034964e-02 5.252879e-02 1.100181e-01 1.741332e-01 4.585693e-01 7.231960e-01
    [57] 9.824822e-01 9.978574e-01 9.982900e-01 9.984829e-01 9.993095e-01 9.993770e-01 9.993912e-01 9.995623e-01 9.998466e-01 9.999271e-01 9.999312e-01 9.999872e-01 9.999891e-01 9.999934e-01
    [71] 9.999996e-01 9.999998e-01 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00 1.000000e+00
    
    $naivebayes$cv$fold5$probs
     [1] 5.128878e-06 5.069363e-06 1.117280e-09 6.579601e-09 9.999872e-01 7.750688e-03 9.995623e-01 1.000000e+00 9.978574e-01 9.982900e-01 1.827851e-06 3.473973e-09 1.191726e-02 2.040910e-06
    [15] 9.824822e-01 4.585693e-01 9.999998e-01 1.572257e-05 1.000000e+00 1.983830e-03 4.239387e-06 2.646258e-05 1.799282e-06 9.993095e-01 1.944447e-06 4.379488e-06 2.315508e-02 7.136240e-06
    [29] 1.501349e-09 5.252879e-02 1.115303e-04 7.231960e-01 7.328366e-05 9.984829e-01 9.999996e-01 1.000000e+00 1.006547e-05 4.034964e-02 1.026141e-04 2.620194e-05 8.355749e-09 7.490103e-03
    [43] 1.000000e+00 1.722777e-03 1.065068e-03 1.041805e-03 9.993770e-01 3.506807e-05 9.999271e-01 1.451176e-04 2.431317e-03 1.100181e-01 9.350814e-05 4.605713e-03 1.741332e-01 1.920745e-06
    [57] 4.539093e-06 7.742514e-08 2.305073e-05 9.998466e-01 1.655608e-06 3.686265e-06 4.760809e-06 5.009503e-06 3.688143e-09 3.168290e-06 9.999891e-01 9.999312e-01 5.640481e-04 2.322587e-05
    [71] 9.999934e-01 1.000000e+00 9.823116e-06 9.993912e-01 5.871301e-06 1.664325e-06
    
    $naivebayes$cv$fold5$labels
     No  No  No  No Yes  No Yes Yes Yes Yes  No  No Yes  No Yes Yes Yes  No Yes  No  No  No  No Yes  No  No  No  No  No  No  No Yes  No Yes Yes Yes  No  No  No  No  No  No Yes  No  No  No Yes  No Yes 
      0   0   0   0   1   0   1   1   1   1   0   0   1   0   1   1   1   0   1   0   0   0   0   1   0   0   0   0   0   0   0   1   0   1   1   1   0   0   0   0   0   0   1   0   0   0   1   0   1 
     No  No Yes  No  No  No  No  No  No  No Yes  No  No  No  No  No  No Yes Yes  No  No Yes Yes  No Yes  No  No 
      0   0   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   1   1   0   0   1   1   0   1   0   0 
    
    $naivebayes$cv$fold5$metrics
    $naivebayes$cv$fold5$metrics$fpr
     [1] 1.00000000 1.00000000 0.98039216 0.96078431 0.94117647 0.92156863 0.90196078 0.88235294 0.86274510 0.84313725 0.82352941 0.80392157 0.78431373 0.76470588 0.74509804 0.72549020 0.70588235
    [18] 0.68627451 0.66666667 0.64705882 0.62745098 0.60784314 0.58823529 0.56862745 0.54901961 0.52941176 0.50980392 0.49019608 0.47058824 0.45098039 0.43137255 0.41176471 0.39215686 0.37254902
    [35] 0.35294118 0.33333333 0.31372549 0.29411765 0.27450980 0.25490196 0.23529412 0.21568627 0.19607843 0.17647059 0.15686275 0.13725490 0.11764706 0.09803922 0.07843137 0.07843137 0.05882353
    [52] 0.03921569 0.01960784 0.01960784 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    [69] 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
    
    $naivebayes$cv$fold5$metrics$tpr
     [1] 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    [39] 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.96 0.96 0.96 0.96 0.92 0.92 0.88 0.84 0.80 0.76 0.72 0.68 0.64 0.60 0.56 0.52 0.48 0.44 0.40 0.36 0.32 0.28 0.24 0.20 0.16 0.12 0.08
    [77] 0.04
    
    
    $naivebayes$cv$fold5$auc
    [1] 0.987451
    
    $naivebayes$cv$fold5$youdins_indx
    [1] 0.1100181
        

</details>

### Impute Incomplete Labeled Data

```R
set.seed(50)

# Introduce some missing data
for (i in 1:ncol(thyroid_data)) {
  thyroid_data[sample(1:nrow(thyroid_data), size = round(nrow(thyroid_data) * .01)), i] <- NA
}

results <- classCV(
  formula = Recurred ~ .,
  data = thyroid_data,
  models = "randomforest",
  train_params = list(
    split = 0.8,
    n_folds = 5,
    stratified = TRUE,
    random_seed = 50,
    standardize = TRUE
  ),
  impute_params = list(method = "impute_bag", args = list(trees = 20, seed_val = 50)),
  model_params = list(final_model = FALSE),
  save = list(models = FALSE, data = FALSE)
)
                   
print(results)
```

**Output**
```
Warning messages:
1: In .clean_data(data, missing_info, !is.null(impute_params$method)) :
  dropping 4 unlabeled observations
2: In .clean_data(data, missing_info, !is.null(impute_params$method)) :
  61 labeled observations are missing data in one or more features and will be imputed

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Random Forest 

Formula: Recurred ~ .

Number of Features: 16

Classes: No, Yes

Training Parameters: list(split = 0.8, n_folds = 5, stratified = TRUE, random_seed = 50, standardize = TRUE, remove_obs = FALSE)

Model Parameters: list(map_args = NULL, final_model = FALSE)

Unlabeled Data: 4

Incomplete Labeled Data: 61

Sample Size (Complete + Imputed Incomplete Labeled Data): 379

Imputation Parameters: list(method = "impute_bag", args = list(trees = 20, seed_val = 50))

Parallel Configs: list(n_cores = NULL, future.seed = NULL)



 Training 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  1.00 

Class:   Precision:  Recall:       F1:

No             1.00     1.00      1.00 
Yes            1.00     0.99      0.99 


 Test 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.94 

Class:   Precision:  Recall:       F1:

No             0.95     0.96      0.95 
Yes            0.90     0.86      0.88 


 Cross-validation (CV) 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.97 ± 0.01 (SD) 

Class:       Average Precision:        Average Recall:            Average F1:

No             0.96 ± 0.01 (SD)       0.99 ± 0.02 (SD)       0.98 ± 0.01 (SD) 
Yes            0.97 ± 0.04 (SD)       0.91 ± 0.03 (SD)       0.94 ± 0.02 (SD) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

Displaying what is contained in the vswift object by converting its class to a list and using R's base `print` function.

```R
class(results) <- "list"
print(results)
```

<details>
    <summary><strong>Output</strong></summary>

    ```
    $configs
    $configs$formula
    Recurred ~ .
    
    $configs$n_features
    [1] 16
    
    $configs$models
    [1] "randomforest"
    
    $configs$model_params
    $configs$model_params$final_model
    [1] FALSE
    
    $configs$model_params$map_args
    NULL
    
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
    [1] TRUE
    
    $configs$train_params$remove_obs
    [1] FALSE
    
    
    $configs$impute_params
    $configs$impute_params$method
    [1] "impute_bag"
    
    $configs$impute_params$args
    $configs$impute_params$args$trees
    [1] 20
    
    $configs$impute_params$args$seed_val
    [1] 50
    
    
    
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
    
    
    
    $missing_data_summary
    $missing_data_summary$unlabeled_data
    [1] 4
    
    $missing_data_summary$incomplete_labeled_data
    [1] 61
    
    $missing_data_summary$complete_data
    [1] 318
    
    
    $class_summary
    $class_summary$classes
    [1] "No"  "Yes"
    
    $class_summary$proportions
    target_vector
           No       Yes 
    0.7176781 0.2823219 
    
    $class_summary$indices
    $class_summary$indices$No
      [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33
     [34]  34  35  36  37  38  39  40  41  42  43  44  45  46  47  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67
     [67]  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  92  93  94  95  96  97  98  99 100 101 102 103 104 105
    [100] 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138
    [133] 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171
    [166] 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204
    [199] 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257
    [232] 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290
    [265] 291 292 293 294 295 337 338 352
    
    $class_summary$indices$Yes
      [1]  48  87  88  89  90  91 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 296 297 298 299 300 301 302
     [34] 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335
     [67] 336 339 340 341 342 343 344 345 346 347 348 349 350 351 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371
    [100] 372 373 374 375 376 377 378 379
    
    
    
    $data_partitions
    $data_partitions$indices
    $data_partitions$indices$split
    $data_partitions$indices$split$train
      [1]  11 101 201  18 279 276 119  58 166  13 104 243 168 277   7  83  69  93  71 173 206   6 255 174 187 214   4  26 151 264  61  51 256
     [34] 135 258  16 216 149 197 186 217  81 352 154 338 242  38  43 111  57 107 132 280 192  78   2  70  44 157 121  92 191 164 241 136  15
     [67] 257 160  96  32 105 141 142 285 265 288  53 137 139  94 126  41 286  65  97 106 266 287  19  30 162  31 167  80 100  56  24 199 134
    [100] 113  95 281 267  10 190 120  66 102 245 158 146 175 209 130 253 138   3 188 124 143 145  17 290 271 127 185 109 177 112 179 207 131
    [133] 294  73 259 152 275 208 210  47 268 212 183 293 181 272 195 200  63 269  12  54  74 278 260   1  60 156  49 205  67 213  33  85 128
    [166]  28 155   9  64 171 284  35   5 292  76   8  34  29 291 270  14  20  79 115 123 125  46 170 117  42 129 198 110 159 196  82 219 153
    [199] 163  98 172  40 116 215 204 148  45  59 261  62 295 220 147  52 244 161 180 305  90 221 227 340 350 344 375 367 234 328 362  48 374
    [232] 336 317 370 314 373 303 233 230  88 355 357 353 327 371 315 356 341 229 307  87 236 223 342 222 349 366 318 239 238 331 330 322 364
    [265] 323 310 339 321 343 313 320 358 335 301 237 309  89 296 297 319 347 365 306 351 332 376 346 361 235 228 311 298 378 334 359 360 326
    [298] 325 312 369 308 224 179 121
    
    $data_partitions$indices$split$test
     [1] 103  50 193  77 182 262  22 211  36 194 108  39 251  25 118  23  84  37 246 144 274 178 250 150  99 189 203 248  68  27 289 165 249 202
    [35] 114 184  72 273 252 133 263 337 122 176 247  21  75  55 140 254 169 283 218  86 324 299 345 368 231 240 379 316  91 304 225 363 377 372
    [69] 300 354 232 302 348 329 333 282 226
    
    
    $data_partitions$indices$cv
    $data_partitions$indices$cv$fold1
     [1] 232 210  50  11 101 201  18 279 276 119  58 166  13 104 243 168 277   7  83  69  93  71 173 206   6 255 174 187 214   4  26 151 264  61
    [35]  51 256 135 258  16 216 149 197 186 217  81 352 154 338 242  38  43 111  57 107 132 280 327 348  87 340 313 237 358 326 299 356 375 377
    [69] 359 229 351 240 362 301 221 222  91
    
    $data_partitions$indices$cv$fold2
     [1] 341 215  76 164 128  67 167 284 170 117 155 268  53 205 267  82 121 130  17   2  27  39 198  40 250 204 106 124  72  32 251 290 163 139
    [35] 118 116  65  15 275 146  84 126 281 193 179 254  41 159  94 169   3 219 152 176 178  24 307 310 336 304 374 357 350 370 302 225 363 354
    [69] 309 233 228 239 346 373 366 372 328
    
    $data_partitions$indices$cv$fold3
     [1] 156 103 138 273  97 208   9  78 108   5  23 127 269   1  12 262  56 259  49  46  20 162 248  10 247 184 131  19  99  28 136 220  34  29
    [35] 150 287  59  79 285 141 295 143  55 244 337 253 112 125  68 213 294 180 183  75 353 224 339 365 316 298 297 342 347 236  89  48 368 230
    [69] 322 361 319 343  88 329 378
    
    $data_partitions$indices$cv$fold4
     [1] 172 265 207  33 145 153  60 123 147  22  63  21 199 120  31 272 114 218  86  85 115 189 241  52 291 177 181 192 182  35 212 140 113  73
    [35]  66  80 266  54  77 270 188 203 286 144 134 160  74 158 109 271  64  96 100  25 234 305 231 308 238 371 334 317 349 325  90 369 226 311
    [69] 355 376 367 223 315 360 318
    
    $data_partitions$indices$cv$fold5
     [1] 194  30 110 293 252 278 171 137 191 185 102 175   8 246 289  42 200  95 196  14 209 133  47  44  45 165 249  62 148 202 288 263 142 129
    [35]  98 260 245  92 122 282  70 292 157 195  36 261 190 161 283 274 211 257 105  37 227 296 324 331 321 303 330 332 335 323 344 320 300 312
    [69] 235 345 379 306 364 314 333
    
    
    
    $data_partitions$proportions
    $data_partitions$proportions$split
    $data_partitions$proportions$split$train
    
           No       Yes 
    0.7203947 0.2796053 
    
    $data_partitions$proportions$split$test
    
           No       Yes 
    0.7142857 0.2857143 
    
    
    $data_partitions$proportions$cv
    $data_partitions$proportions$cv$fold1
    
           No       Yes 
    0.7142857 0.2857143 
    
    $data_partitions$proportions$cv$fold2
    
           No       Yes 
    0.7142857 0.2857143 
    
    $data_partitions$proportions$cv$fold3
    
      No  Yes 
    0.72 0.28 
    
    $data_partitions$proportions$cv$fold4
    
      No  Yes 
    0.72 0.28 
    
    $data_partitions$proportions$cv$fold5
    
      No  Yes 
    0.72 0.28 
    
    
    
    
    $metrics
    $metrics$randomforest
    $metrics$randomforest$split
           Set Classification Accuracy Class: No Precision Class: No Recall Class: No F1 Class: Yes Precision Class: Yes Recall Class: Yes F1
    1 Training               0.9967105           0.9954545        1.0000000    0.9977221            1.0000000         0.9882353     0.9940828
    2     Test               0.9350649           0.9464286        0.9636364    0.9549550            0.9047619         0.8636364     0.8837209
    
    $metrics$randomforest$cv
                        Fold Classification Accuracy Class: No Precision Class: No Recall Class: No F1 Class: Yes Precision Class: Yes Recall
    1                 Fold 1             0.961038961         0.948275862      1.000000000  0.973451327           1.00000000        0.86363636
    2                 Fold 2             0.974025974         0.964912281      1.000000000  0.982142857           1.00000000        0.90909091
    3                 Fold 3             0.973333333         0.981481481      0.981481481  0.981481481           0.95238095        0.95238095
    4                 Fold 4             0.973333333         0.964285714      1.000000000  0.981818182           1.00000000        0.90476190
    5                 Fold 5             0.946666667         0.962962963      0.962962963  0.962962963           0.90476190        0.90476190
    6               Mean CV:             0.965679654         0.964383660      0.988888889  0.976371362           0.97142857        0.90692641
    7 Standard Deviation CV:             0.011935749         0.011769708      0.016563466  0.008327711           0.04259177        0.03144121
    8     Standard Error CV:             0.005337829         0.005263573      0.007407407  0.003724266           0.01904762        0.01406094
      Class: Yes F1
    1   0.926829268
    2   0.952380952
    3   0.952380952
    4   0.950000000
    5   0.904761905
    6   0.937270616
    7   0.021121788
    8   0.009445951
    ```
</details>

### Using Parallel Processing

Parallel processing operates at the fold level, which means the system can simultaneously process multiple cross-validation folds (and the train-test split) even when training a single model.

*Note*: This example uses the [Internet Advertisements data from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/51/internet+advertisements).

```R
set.seed(NULL)

# Set url for Internet Advertisements data from UCI Machine Learning Repository. This data has 3,278 instances and 1558 features.
url <- "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip"

# Set file destination
dest_file <- file.path(getwd(), "ad.zip")

# Download zip file
download.file(url, dest_file)

# Unzip file
unzip(zipfile = dest_file, files = "ad.data")

# Read data
ad_data <- read.csv("ad.data")

# Load in vswift
library(vswift)

# Create arguments variable to tune parameters for multiple models
map_args <- list(
  "knn" = list(ks = 5),
  "xgboost" = list(
    params = list(
      booster = "gbtree",
      objective = "reg:logistic",
      lambda = 0.0003,
      alpha = 0.0003,
      eta = 0.8,
      max_depth = 6
    ),
    nrounds = 10
  )
)

print("Without Parallel Processing:")

# Obtain new start time
start <- proc.time()

# Run the same model without parallel processing
results <- classCV(
  data = ad_data,
  target = "ad.",
  models = c("knn", "svm", "decisiontree", "xgboost"),
  train_params = list(
    split = 0.8,
    n_folds = 5,
    random_seed = 50
  ),
  model_params = list(map_args = map_args)
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
results <- classCV(
  data = ad_data,
  target = "ad.",
  models = c("knn", "svm", "decisiontree", "xgboost"),
  train_params = list(
    split = 0.8,
    n_folds = 5,
    random_seed = 50
  ),
  model_params = list(map_args = map_args),
  parallel_configs = list(
    n_cores = 6,
    future.seed = 100
  )
)

# Obtain end time
end_par <- proc.time() - start_par

# Print time
print(end_par)
```

**Output**
```
[1] "Without Parallel Processing:"

Warning message:
In .create_dictionary(preprocessed_data[, vars$target]) :
  creating keys for target variable due to 'logistic' or 'xgboost' being specified;
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
 227.53    4.24  214.43

[1] "Parallel Processing:"

Warning message:
In .create_dictionary(preprocessed_data[, vars$target]) :
  creating keys for target variable due to 'logistic' or 'xgboost' being specified;
  classes are now encoded: ad. = 0, nonad. = 1

   user  system elapsed 
   1.64    4.67   96.27
```

```R
# Print parameter information and model evaluation metrics; If number of features > 20, the target replaces the formula
print(results, models = c("xgboost", "knn"))
```

**Output**
```
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: Extreme Gradient Boosting 

Target: ad.

Number of Features: 1558

Classes: ad., nonad.

Training Parameters: list(split = 0.8, n_folds = 5, random_seed = 50, stratified = FALSE, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = list(xgboost = list(params = list(booster = "gbtree", objective = "reg:logistic", lambda = 3e-04, alpha = 3e-04, eta = 0.8, max_depth = 6), nrounds = 10)), logistic_threshold = 0.5, final_model = FALSE)

Unlabeled Data: 0

Incomplete Labeled Data: 0

Sample Size (Complete Data): 3278

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = 6, future.seed = 100)



 Training 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.99 

Class:      Precision:  Recall:       F1:

ad.               0.99     0.93      0.96 
nonad.            0.99     1.00      0.99 


 Test 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.98 

Class:      Precision:  Recall:       F1:

ad.               0.97     0.89      0.93 
nonad.            0.98     0.99      0.99 


 Cross-validation (CV) 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.98 ± 0.01 (SD) 

Class:          Average Precision:        Average Recall:            Average F1:

ad.               0.95 ± 0.02 (SD)       0.88 ± 0.04 (SD)       0.91 ± 0.03 (SD) 
nonad.            0.98 ± 0.01 (SD)       0.99 ± 0.00 (SD)       0.99 ± 0.00 (SD) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


Model: K-Nearest Neighbors 

Target: ad.

Number of Features: 1558

Classes: ad., nonad.

Training Parameters: list(split = 0.8, n_folds = 5, random_seed = 50, stratified = FALSE, standardize = FALSE, remove_obs = FALSE)

Model Parameters: list(map_args = list(knn = list(ks = 5)), logistic_threshold = 0.5, final_model = FALSE)

Unlabeled Data: 0

Incomplete Labeled Data: 0

Sample Size (Complete Data): 3278

Imputation Parameters: list(method = NULL, args = NULL)

Parallel Configs: list(n_cores = 6, future.seed = 100)



 Training 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  1.00 

Class:      Precision:  Recall:       F1:

ad.               1.00     0.99      1.00 
nonad.            1.00     1.00      1.00 


 Test 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Classification Accuracy:  0.96 

Class:      Precision:  Recall:       F1:

ad.               0.89     0.80      0.84 
nonad.            0.97     0.98      0.98 


 Cross-validation (CV) 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

Average Classification Accuracy:  0.93 ± 0.01 (SD) 

Class:          Average Precision:        Average Recall:            Average F1:

ad.               0.71 ± 0.07 (SD)       0.82 ± 0.01 (SD)       0.76 ± 0.04 (SD) 
nonad.            0.97 ± 0.00 (SD)       0.95 ± 0.02 (SD)       0.96 ± 0.01 (SD) 

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

```R
# Plot results
plot(
  results,
  models = "xgboost",
  class_names = "ad.",
  metrics = c("precision", "recall"),
  path = getwd()
)
```

<details>
  
  <summary><strong>Plots</strong></summary>

  ![image](assets/ads/extreme_gradient_boosting_cv_precision_ad..png)
  ![image](assets/ads/extreme_gradient_boosting_cv_recall_ad..png)
  ![image](assets/ads/extreme_gradient_boosting_train_test_precision_ad..png)
  ![image](assets/ads/extreme_gradient_boosting_train_test_recall_ad..png)

</details>

## Acknowledgements
The development of this package was inspired by other machine learning packages such as
topepo's [caret](https://github.com/topepo/caret) package and the
[scikit-learn](https://github.com/scikit-learn/scikit-learn) package.
