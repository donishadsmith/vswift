# Changelog

All notable future changes to vswift will be documented in this file.

## [Versioning Notice]

**As this package is still in the version 0.x.x series, aspects of the package may change rapidly to improve convenience and ease of use.**

**Additionally, beyond version 0.1.1, versioning for the 0.x.x series for this package will work as:**

`0.minor.patch`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (i.e new functions or parameters, changes in parameter defaults or function names, etc).
- *.patch* : Contains no new features, simply fixes any identified bugs.


## [0.2.0.9000] - 2024-09-09 [Development Version; RE-UPLOAD]
### ♻ Changed
- Refactored package internally to make code more reusable and maintainable. Version is in still in development but has
passed previous testthat tests and all functions can be used.
- Some parameters for `classCV` and `genFolds` have been grouped together. For instance, `split`, `n_folds`, `standardize`,
`remove_obs`, `random_seed`, `stratified`, etc are no longer separate input parameters. They are now apart of the new
`train_params` parameter as elements (e.g `train_params = list(split = 0.8, n_folds = 5, standardize = TRUE))`. Additionally,
`model_params`, `save`, and `parallel_configs` were also created to group similar parameters in the former `classCV`
input parameters. For all functions, `model_type` is not `models` and for `print`, `parameters` is now `configs`.
- `classCV` output object is more organized and includes "configs" for user specified arguments and model-specific arguments,
"class_summary" for information pertaining to classes such as the names of the classes, indices, proportions,
"data_partitions" to include the indices, class proportions in each split/fold, and dataframes if requested, "imputation"
for imputation information, "models" if models are requested, and "metrics" for metrics.
- Can request final model with having to specify `n_folds` or `split`.
### 🐛 Fixes
- The previous behavior were observations that are missing the target or excluded; however in addition to this,
when imputation is requested, the target variable is excluded from being a predictor for imputation.
- Prior to imputation, regardless if standardization is requested, all numerical columns are standardized.
- Error when saving plots in RStudio.
- Metrics for latest GBM version should no longer produce NAs.
- [RE-UPLOAD]: Version that allows final model to run without specifying `split` or `n_folds`,
also includes additional tests. Still has the same version number - 0.2.0.9000. Re-upload also includes a fix for
logistic regression, since `model_params$threshold` was called instead of `model_params$logistic_threshold`, resulting in
the logistic threshold to be NULL and the prediction vector empty.

## [0.1.4] - 2024-08-08 [RE-UPLOAD]
### 🐛 Fixes
- Added `plan(sequential)` so background workers don't stay up and continue to consume RAM.

## [0.1.4] - 2024-08-07
### ♻ Changed
- Framework for parallel processing changed from doParallel to future, specifically, futures multisession is used.
### 🐛 Fixes
- Switch from doParallel to future fixes parallel processing on Ubuntu, where it used to freeze during parallel processing.
- set.seed is now also set in the internal function .generate_model to use certain seeds for models, such as neural network,
that have some stochastic elements.

## [0.1.3] - 2024-07-29
### 🐛 Fixes
- Fix issue with `classCV` output list not storing information about the missing data in each column
for both the training and validation set used for "split" and for each cv fold.

## [0.1.2] - 2024-06-20
### 🐛 Fixes
- kknn's `contr.dummy` not being found as a function.

## [0.1.1] - 2024-06-19
### ♻ Changed
- Changed order of parameters for `classCV` function.

### 🐛 Fixes
- Standardizes validation data using the mean and standard deviation of the training set.

### 💻 Metadata
- Improved documentation.

## [0.1.0] - 2024-05-13
- First release of vswift package