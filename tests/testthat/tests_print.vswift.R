library(vswift)
library(testthat)

test_that("testing print function", {
  data <- iris

  args <- list("nnet" = list(size = 10))

  expect_no_error(result <- classCV(
    data = data, target = 5, models = c("randomforest", "nnet", "svm"),
    train_params = list(split = 0.8, n_folds = 5, remove_obs = T, stratified = T),
    model_params = list(map_args = args), save = list(models = T, data = T)
  ))
  expect_no_error(
    print(result, models = c("svm", "nnet"))
  )
  expect_no_error(
    print(result)
  )
  expect_no_error(
    print(result, models = c("svm", "nnet"), metrics = T)
  )
})
