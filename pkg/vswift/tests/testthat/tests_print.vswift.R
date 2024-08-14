library(vswift)
library(testthat)

test_that("testing print function", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "ann" = list(size = 10))
  
  expect_no_error(result <- classCV(data = data, target = 5, models = c("knn", "randomforest","ann","svm"), 
                                    train_params = list(split = 0.8, n_folds = 5, remove_obs = T, stratified = T),
                                    model_params = list(map_args = args), save = list(models = T, data = T)))
  expect_no_error(
    print(result, models = c("knn","ann"))
    )
  expect_no_error(
    print(result)
  )
  expect_no_error(
    print(result, models = c("knn","ann"), metrics = T)
  )
  }
)