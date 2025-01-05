library(vswift)
library(testthat)

test_that("testing if split and cv works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, train_params = list(split = 0.8, n_folds = 5, stratified = T), create_data = T)
  )
  expect_no_error(
    folds <- genFolds(data = data, target = "Species", train_params = list(split = 0.8, n_folds = 5, stratified = F), create_data = T)
  )
  expect_true(is.data.frame(folds$data_partitions$dataframes$cv$fold1$train))
  expect_true(is.data.frame(folds$data_partitions$dataframes$split$train))
})

test_that("testing if split works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, train_params = list(split = 0.8, stratified = T), create_data = T)
  )
  expect_no_error(
    folds <- genFolds(data = data, target = 5, train_params = list(split = 0.8, stratified = F), create_data = T)
  )

  expect_true(!is.null(folds))
})


test_that("testing if cv works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, train_params = list(n_folds = 5, stratified = T), create_data = T)
  )
  expect_no_error(
    folds <- genFolds(data = data, target = 5, train_params = list(n_folds = 5, stratified = F), create_data = T)
  )

  expect_true(!is.null(folds))
})
