library(vswift)
library(testthat)

test_that("testing if split and cv works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, split = 0.8, n_folds = 5, stratified = T, create_data = T)
    )
  expect_no_error(
    folds <- genFolds(data = data, target = 5, split = 0.8, n_folds = 5, stratified = F, create_data = T)
  )
  expect_true(is.data.frame(folds[["data"]][["cv"]][["fold 1"]]))
}
)

test_that("testing if split works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, split = 0.8, stratified = T, create_data = T)
  )
  expect_no_error(
    folds <- genFolds(data = data, target = 5, split = 0.8, stratified = F, create_data = T)
  )
}
)


test_that("testing if cv works for genFolds", {
  data <- iris
  expect_no_error(
    folds <- genFolds(data = data, target = 5, n_folds = 5, stratified = T, create_data = T)
  )
  expect_no_error(
    folds <- genFolds(data = data, target = 5, n_folds = 5, stratified = F, create_data = T)
  )
}
)

