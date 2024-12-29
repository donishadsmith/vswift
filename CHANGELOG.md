# Changelog

All notable future changes to vswift will be documented in this file.

## [Versioning Notice]

**As this package is still in the version 0.x.x series, aspects of the package may change rapidly to improve convenience and ease of use.**

**Additionally, beyond version 0.1.1, versioning for the 0.x.x series for this package will work as:**

`0.minor.patch`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (i.e new functions or parameters, changes in parameter defaults or function names, etc).
- *.patch* : Contains no new features, simply fixes any identified bugs.

## [0.4.0.9002] - 2024-12-28
### 📖 Documentation
- Name changes in documentation
### ♻ Changed
- For the plot title, "Logistic Regression" changed to "Unregularized Logistic Regression".
- For the plot title, "Multinomial Logistic Regression" changed to "Unregularized Multinomial Logistic Regression".

## [0.4.0.9002] - 2024-12-28
### 🚀 New/Added
- Added two new models: "regularized_logistic" & "regularized_multinomial" for penalized logistic regression and
multinomial logistic regression with glmnet.
### ♻ Changed
- Cleaned/removed some additional arguments.
- For the plot title, "Logistic Regression" changed to "Un-Regularized Logistic Regression".
### 🐛 Fixes
- Fixed a warning message in stratified sampling that was partially missing some information.

## [0.4.0.9001] - 2024-12-27
### 🐛 Fixes
- Internal refactoring to not create all partitions prior to validation to reduce overhead and improve resource utilization especially for parallel processing.
### ♻ Changed
- In vswift object, `imputation` changed to `imputation_models`.
- Changed name of imputation methods: 'bag_impute' -> 'impute_bag' and 'knn_impute' -> 'impute_knn'.

## [0.4.0.9000] - 2024-12-26
### ♻ Changed
- `save_plots` parameter removed in `plot` function, now `path` must be used to save plots.
- The parameter order in `plot` has been changed to improve organization.
- Documentation has been improved.

## [0.3.0] - 2024-12-18
### ♻ Changed
- Renamed models for clarity: "ann" -> "nnet" and "gbm" -> "xgboost".
- Changed "F-score" to "F1".
- Enhanced output formatting: the standard deviation values now clearly labeled next to mean values in cross-validation results.
- Improved warning message clarity by: 1) adding distinct warnings for invalid model names in `models` parameter and `model_params$map_args` and 2) adding warning about standardizing prior to imputation when `train_params$standardize` is `FALSE`.
- Added legend to plots that distinguishes between mean and standard deviation lines in cross-validation visualizations.

### 🐛 Fixes
- Error with standardizing prior to creating the final model due to some columns not being of class numeric. This issue did not
affect the training and test data only the data used to create the final model.

## [0.2.7] - 2024-12-08
### 🐛 Fixes
- Change warnings to errors when split, n_folds, and final_model is not used and fix related conditions.
- Resolve issue where missing targets are not removed when the target variable is NULL.
- Add missing data summary to classCV output and differentiate between unlabeled and labeled observations with missing data in the printed output.
- Improve handling of leftovers during train-test splitting using stratified sampling.
- Fix issue where not all cross-validation plots are produced and plots open on a separate device in RStudio.

## [0.2.6] - 2024-12-04
### 🐛 Fixes
- Clean namespace
- Add error when `split`, `n_folds`, or `final_model` is not set

## [0.2.5] - 2024-11-25
- Small update to allow `skip` has passed parameter for "ann" model.

## [0.2.4] - 2024-11-08
### 🐛 Fixes
- Fixes logistic decision rule such that probabilities of 0.50 are assigned to the positive instance
instead of negative instance. Previous rule followed `"1" if P(Class = 1 | Features) > threshold; "0" otherwise` now follows `"1" if P(Class = 1 | Features) >= threshold; "0" otherwise`.

## [0.2.3] - 2024-10-28
### 🐛 Fixes
- Ensures internal scaling for multiple models are not done so that scaling is only handled by the `scale` parameter
for `classCV`.
- For "gbm", no more warning about using "formula".

## [0.2.2] - 2024-10-27
### 🐛 Fixes
- For "logistic", fixed error that occurs when using `formula` instead of `target` due to the target in the formula
not being extracted when assessing if the target variable is binary.
### ♻ Changed
- If "gbm" has a logistic regression objective function, then if the target is not binary, it will fail faster.

## [0.2.1] - 2024-10-23
### 🐛 Fixes
- For "gbm", only the "multi:softmax" and "binary:hinge" objective could be used since these objectives print labels directly
and didn't need to be converted from probability or logit to labels. Now, the following objectives can be
used: "reg:logistic", "binary:logistic", "binary:logitraw", "binary:hinge", and "multi:softprob". Additionally,
`logistic_threshold` is used if the following objectives are used for "gbm": "reg:logistic", "binary:logistic",
and "binary:logitraw" objectives are used for.

## [0.2.0] - 2024-09-15
- Non-Development version of 0.2.0.
- Minor internal code changes.
- Additional testthat tests to assess outputs.
- Removes explicit roxygen2 export of internal private functions.
- Documents the `contr.dummy` function from the `kknn` package since `train.kknn` requires this function to be in the
current namespace to work.
- Allows additional arguments for multiple algorithms to be used.

## [0.2.0.9000] - 2024-09-09 [Development Version; RE-UPLOAD]
### ♻ Changed
- Refactored package internally to make code more reusable and maintainable. Version is in still in development but has
passed previous testthat tests and all functions can be used. Commit [here](https://github.com/donishadsmith/vswift/commit/5c06e40eadf57ef610dc5648d87db78160f211e2).
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
