# Changelog

All notable future changes to vswift will be documented in this file.

## [Versioning Notice]

**As this package is still in the version 0.x.x series, aspects of the package may change rapidly to improve convenience and ease of use.**

**Additionally, beyond version 0.1.1, versioning for the 0.x.x series for this package will work as:**

`0.minor.patch`

- *.minor* : Introduces new features and may include potential breaking changes. Any breaking changes will be explicitly
noted in the changelog (i.e new functions or parameters, changes in parameter defaults or function names, etc).
- *.patch* : Contains no new features, simply fixes any identified bugs.

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