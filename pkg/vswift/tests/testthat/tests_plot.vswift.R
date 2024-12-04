library(vswift)
library(testthat)

test_that("testing plot function", {
  data <- iris

  args <- list("knn" = list(ks = 3), "ann" = list(size = 10))

  expect_no_error(result <- classCV(
    data = data, target = 5, models = c("knn", "randomforest", "ann", "svm"),
    train_params = list(split = 0.8, n_folds = 5, remove_obs = T, stratified = T),
    model_params = list(map_args = args), save = list(models = T, data = T)
  ))

  expect_no_error(
    plot(result, models = "knn", split = T, cv = T, class_names = "setosa")
  )

  expect_no_error(
    plot(result, models = "knn", split = T, cv = T, class_names = "setosa", save_plots = TRUE)
  )
})
