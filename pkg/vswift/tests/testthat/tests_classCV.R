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
  if(requireNamespace("mlbench", quietly = TRUE)){
    data("PimaIndiansDiabetes")
    # Create binary
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
  if(requireNamespace("mlbench", quietly = TRUE)){
    data("PimaIndiansDiabetes")
    # Create binary
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
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("train-test split and k-fold CV for all models", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "ann", size = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic", stratified = TRUE))
  } else {
    skip("mlbench package not available")
  }
})

