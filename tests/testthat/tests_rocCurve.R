library(vswift)
library(testthat)

source("utils.R")

# Test that each model works with train-test splitting alone
test_that("test roc curve", {
  data <- iris

  data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
  data$Species <- factor(data$Species)

  map_args <- list(
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
    ),
    "regularized_logistic" = list(alpha = 1, nfolds = 3),
    "nnet" = list(size = 2)
  )

  models <- names(vswift:::.MODEL_LIST)[!names(vswift:::.MODEL_LIST) == "knn"]

  results <- classCV(
    formula = Species ~ .,
    data = data,
    models = models,
    model_params = list(map_args = map_args, rule = "1se", verbose = TRUE),
    train_params = list(
      split = 0.8,
      n_folds = 5,
      standardize = T,
      stratified = TRUE,
      random_seed = 123
    ),
    save = list(models = TRUE, data = TRUE)
  )

  # With thresholds derived from models
  roc_output <- rocCurve(results, path = getwd())
  check_png()
  expect_true(length(roc_output) == "12")
  check_metrics(roc_output, "roc")

  # With specified thresholds
  roc_output <- rocCurve(results, path = getwd(), thresholds = seq(0, 0.9, 0.1))
  check_png()
  expect_true(length(roc_output) == "12")
  check_metrics(roc_output, "roc")
})

test_that("test equivalence with standardizing", {
  data <- iris

  data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
  data$Species <- factor(data$Species)

  result1 <- classCV(
    formula = Species ~ .,
    data = data,
    models = "svm",
    train_params = list(
      split = 0.8,
      n_folds = 5,
      standardize = T,
      stratified = TRUE,
      random_seed = 123
    ),
    save = list(models = TRUE, data = TRUE)
  )

  output1 <- rocCurve(result1)

  result2 <- classCV(
    formula = Species ~ .,
    data = data,
    models = "svm",
    train_params = list(
      split = 0.8,
      n_folds = 5,
      standardize = T,
      stratified = TRUE,
      random_seed = 123
    ),
    save = list(models = TRUE, data = TRUE)
  )

  output2 <- rocCurve(result2, data)

  for (fold in names(output1$svm$cv)) {
    for (i in names(output1$svm$cv[[fold]])) {
      if (i == "metrics") {
        expect_true(all(output1$svm$cv[[fold]]$metrics$tpr == output2$svm$cv[[fold]]$metrics$tpr))
        expect_true(all(output1$svm$cv[[fold]]$metrics$fpr == output2$svm$cv[[fold]]$metrics$fpr))
      } else {
        expect_true(all(output1$svm$cv[[fold]][[i]] == output2$svm$cv[[fold]][[i]]))
      }
    }
  }
})


test_that("test equivalence with imputation", {
  data <- iris

  data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
  data$Species <- factor(data$Species)

  set.seed(123)

  # Introduce some missing data
  for (i in 1:ncol(data)) {
    data[sample(1:nrow(data), size = round(nrow(data) * .01)), i] <- NA
  }

  result1 <- classCV(
    formula = Species ~ .,
    data = data,
    models = "svm",
    train_params = list(
      split = 0.8,
      n_folds = 5,
      stratified = TRUE,
      random_seed = 123,
      standardize = TRUE
    ),
    impute_params = list(method = "impute_bag", args = list(trees = 20, seed_val = 123)),
    save = list(models = TRUE, data = TRUE)
  )

  output1 <- rocCurve(result1)

  result2 <- classCV(
    formula = Species ~ .,
    data = data,
    models = "svm",
    train_params = list(
      split = 0.8,
      n_folds = 5,
      stratified = TRUE,
      random_seed = 123,
      standardize = TRUE
    ),
    impute_params = list(method = "impute_bag", args = list(trees = 20, seed_val = 123)),
    save = list(models = TRUE)
  )

  output2 <- rocCurve(result2, data)

  for (fold in names(output1$svm$cv)) {
    for (i in names(output1$svm$cv[[fold]])) {
      if (i == "metrics") {
        expect_true(all(output1$svm$cv[[fold]]$metrics$tpr == output2$svm$cv[[fold]]$metrics$tpr))
        expect_true(all(output1$svm$cv[[fold]]$metrics$fpr == output2$svm$cv[[fold]]$metrics$fpr))
      } else {
        expect_true(all(output1$svm$cv[[fold]][[i]] == output2$svm$cv[[fold]][[i]]))
      }
    }
  }
})
