library(vswift)
library(testthat)
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
      random_seed = 50
    ),
    save = list(models = TRUE, data = TRUE)
  )

  # With thresholds derived from models
  output <- rocCurve(results, path = getwd())

  for (png_file in list.files(getwd(), pattern = ".png")) {
    expect_true(file.size(png_file) > 0)
    file.remove(png_file)
  }

  file.remove(list.files(getwd(), pattern = "Rplots.pdf"))

  expect_true(length(output) == "13")

  # With specified thresholds
  output <- rocCurve(results, path = getwd(), thresholds = seq(0, 0.9, 0.1))

  for (png_file in list.files(getwd(), pattern = ".png")) {
    expect_true(file.size(png_file) > 0)
    file.remove(png_file)
  }

  file.remove(list.files(getwd(), pattern = "Rplots.pdf"))

  expect_true(length(output) == "13")
})
