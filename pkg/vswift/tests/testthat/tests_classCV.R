# Testing classCV function
library(vswift)
# Test that each model works with train-test splitting alone
test_that("test train-test split for all models and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "ann", size = 10))
  if(requireNamespace("mlbench", quietly = TRUE)){
    # Create binary
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("test train-test split for all models with stratified sampling", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "ann", size = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    # Create binary
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("k-fold CV for all models no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "ann", size = 10))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("k-fold CV for all models", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "ann", size = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("train-test split and k-fold CV for all models", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "ann", size = 10))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("test imputation and missing data", {
  
  data <- iris
  # Introduce NA
  data <- missForest::prodNA(data)
  # randomforest
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE), model_type = "lda", stratified = TRUE))
  
  # simple
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "simple", model_type = "lda", stratified = TRUE))
  
  # complete cases only
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE))
})

test_that("test imputations", {
  if(requireNamespace("missForest", quietly = TRUE)){
    data <- iris
    # Introduce NA
    data <- missForest::prodNA(data)
    # randomforest no impute arguments
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", model_type = "lda", stratified = TRUE))
    # randomforest with impute arguments
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE, maxiter = 1000, maxnodes = 5), model_type = "lda", stratified = TRUE, final_model = TRUE))
    # random forest with incorrect impute arguments
    expect_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE, maxiter = 1000, try = 5), model_type = "lda", stratified = TRUE, final_model = TRUE))
    # simple
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "simple", model_type = "lda", stratified = TRUE, final_model = TRUE))
    # complete cases only
    expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE, final_model = TRUE))

  } else {
    skip("missForest package not available")
  }})

test_that("test saving features", {
  data <- iris
  
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE,
                                    save_models = TRUE, save_data = TRUE))
  
})

test_that("test input errors", {
  
  data <- iris
  
  expect_error(result <- classCV(target = "Species", split = 0.1, model_type = "lda"))
  # split out of range
  expect_error(result <- classCV(data = data, target = "Species", split = 0.1, model_type = "lda"))
  # n_folds out of range
  expect_error(result <- classCV(data = data, target = "Species", n_folds = 31, model_type = "lda"))
  # model not specified
  expect_error(result <- classCV(data = data, target = "Species", n_folds = 31, model_type = "lda"))
  # no target
  expect_error(result <- classCV(data = data, n_folds = 31, model_type = "lda"))
  # target out of range
  expect_error(result <- classCV(data = data, target = 5, n_folds = 31, model_type = "lda"))
  # target also predictor
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, model_type = "lda"))
  # incorrect imputation
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, impute_method = "knn", model_type = "lda"))
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, model_type = "cnn"))
})


