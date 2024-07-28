# Testing classCV function
library(vswift)
library(testthat)
# Test that each model works with train-test splitting alone
test_that("test train-test split and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "lda"))
})

test_that("test new formula method", {
  data <- iris
  result1 <- classCV(data = data, target = "Species", split = 0.8, model_type = "qda", random_seed = 123)
  expect_no_error(result2 <- classCV(formula = Species ~ ., data = data, split = 0.8, model_type = "qda",
                                     random_seed = 123))
  expect_equal(result1$metrics$lda$split,result2$metrics$lda$split)
})

test_that("k-fold CV no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "svm"))
})


test_that("k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "ann", size = 10,
                                    stratified = TRUE, random_seed = 123))
})


test_that("train-test split and k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3,
                                    model_type = "naivebayes", stratified = TRUE))
})

test_that("train-test split and k-fold CV without stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "multinom"))
})


test_that("test imputation and missing data", {
  data <- iris
  # Introduce NA
  for(i in 1:ncol(data)){
    data[sample(1:nrow(data), size = round(nrow(data)*.10)),i] <- NA
  }
  
  # knn
  expect_warning(expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3,
                                                  impute_method = "knn_impute", impute_args = list(neighbors = 5),
                                                  model_type = "qda", stratified = TRUE)))
  
  # bag
  expect_warning(expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3,
                                                  impute_method = "bag_impute", impute_args = list(trees = 5),
                                                  model_type = "multinom", stratified = TRUE)))
  
  # complete cases only
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3,
                                   model_type = "decisiontree", stratified = TRUE))
})

test_that("test random seed", {
  
  data <- iris
  result_1 <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", random_seed = 123,
                      stratified = TRUE, ks = 5)
  result_2 <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", random_seed = 123,
                      stratified = TRUE, ks = 5)
  
  expect_equal(result_1$sample_indices$split$training,result_2$sample_indices$split$training)
  expect_equal(result_1$metrics$cv,result_2$metrics$cv)
})

test_that("running multiple models", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,
                                                                max_depth = 6), nrounds = 10))
  
  expect_warning(result <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "svm", "gbm", "randomforest"), 
                                   n_folds = 5, save_data = T, mod_args = args, save_models = T, remove_obs = T,
                                   stratified = T, random_seed = 123))
  
  class(data$Species) <- "character"
  
  data$Species <- ifelse(data$Species == 1, 0, 1)
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(objective = "multi:softprob",num_class = 2,
                                                                eta = 0.3,max_depth = 6), nrounds = 10))
  
  expect_no_error(result <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "svm", "gbm", "randomforest"), 
                                   n_folds = 5, save_data = T, mod_args = args, save_models = T, remove_obs = T,
                                   stratified = T, random_seed = 123, standardize = TRUE))
})

test_that("n_cores", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,
                                                                max_depth = 6), nrounds = 10))
  
  expect_warning(result1 <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "svm", "gbm", "randomforest"), 
                                   n_folds = 5, save_data = T, mod_args = args, save_models = T, remove_obs = T,
                                   stratified = T, random_seed = 123, n_cores = 2))
  
  expect_warning(result2 <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "svm", "gbm", "randomforest"), 
                                    n_folds = 5, save_data = T, mod_args = args, save_models = T, remove_obs = T,
                                    stratified = T, random_seed = 123))
  
  expect_equal(result1$metrics$knn$folds,result2$metrics$knn$folds)
  
})