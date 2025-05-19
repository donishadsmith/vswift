library(vswift)
library(testthat)

test_that("testing plot function", {
  data <- iris

  args <- list("nnet" = list(size = 10))

  expect_no_error(result <- classCV(
    data = data, target = 5, models = c("randomforest", "nnet", "svm"),
    train_params = list(split = 0.8, n_folds = 5, remove_obs = T, stratified = T),
    model_params = list(map_args = args), save = list(models = T, data = T)
  ))

  expect_no_error(
    plot(result, models = "nnet", split = T, cv = T, class_names = "setosa")
  )

  expect_no_error(
    plot(result, models = "nnet", class_names = "setosa", path = getwd())
  )

  for (png_file in list.files(getwd(), pattern = ".png")) {
    expect_true(file.size(png_file) > 0)
    file.remove(png_file)
  }

  file.remove(list.files(getwd(), pattern = "Rplots.pdf"))
})
