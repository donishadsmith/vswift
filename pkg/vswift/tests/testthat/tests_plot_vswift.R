library(vswift)
library(testthat)

test_that("testing print function", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "ann" = list(size = 10))
  
  expect_no_error(result <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "randomforest","ann","svm"), 
                                    n_folds = 5, mod_args = args, save_data = T, save_models = T, remove_obs = T, stratified = T))
  
  expect_no_error(
    plot(result, model_type = c("knn"), split = T, cv = F)
  )
}
)