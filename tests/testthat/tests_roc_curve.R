library(vswift)
library(testthat)

source("utils.R")

# Test that each model works with train-test splitting alone
test_that("test roc curve", {
  data <- iris

  data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
  data$Species <- factor(data$Species)

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
    ),
    "regularized_logistic" = list(alpha = 1, nfolds = 3),
    "nnet" = list(size = 2)
  )

  models <- c(
    "regularized_logistic", "regularized_multinomial", "multinom", "knn", "nnet", "lda", "qda",
    "svm", "decisiontree", "randomforest", "logistic", "naivebayes", "xgboost"
  )

  results <- class_cv(
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
  roc_output <- results$roc_curve(path = getwd())
  check_png()
  expect_true(length(roc_output$get_model()) == 13)
  check_metrics(roc_output, "roc")

  # With specified thresholds
  roc_output <- results$roc_curve(path = getwd(), thresholds = seq(0, 0.9, 0.1))
  check_png()
  expect_true(length(roc_output$get_model()) == 13)
  check_metrics(roc_output, "roc")
})

test_that("test equivalence with standardizing", {
  data <- iris

  data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
  data$Species <- factor(data$Species)

  result1 <- class_cv(
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

  output1 <- result1$roc_curve()

  result2 <- class_cv(
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

  output2 <- result2$roc_curve(data = data)

  svm1 <- output1$get_model("svm")
  svm2 <- output2$get_model("svm")

  for (fold in names(svm1$cv)) {
    for (i in names(svm1$cv[[fold]])) {
      if (i == "metrics") {
        expect_true(all(svm1$cv[[fold]]$metrics$tpr == svm2$cv[[fold]]$metrics$tpr))
        expect_true(all(svm1$cv[[fold]]$metrics$fpr == svm2$cv[[fold]]$metrics$fpr))
      } else {
        expect_true(all(svm1$cv[[fold]][[i]] == svm2$cv[[fold]][[i]]))
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

  result1 <- class_cv(
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

  output1 <- result1$roc_curve()

  result2 <- class_cv(
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

  output2 <- result2$roc_curve(data = data)

  svm1 <- output1$get_model("svm")
  svm2 <- output2$get_model("svm")

  for (fold in names(svm1$cv)) {
    for (i in names(svm1$cv[[fold]])) {
      if (i == "metrics") {
        expect_true(all(svm1$cv[[fold]]$metrics$tpr == svm2$cv[[fold]]$metrics$tpr))
        expect_true(all(svm1$cv[[fold]]$metrics$fpr == svm2$cv[[fold]]$metrics$fpr))
      } else {
        expect_true(all(svm1$cv[[fold]][[i]] == svm2$cv[[fold]][[i]]))
      }
    }
  }
})
