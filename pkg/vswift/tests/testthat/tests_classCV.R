# Testing classCV function
library(vswift)
library(testthat)
# Test that each model works with train-test splitting alone
test_that("test train-test split and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "lda", train_params = list(split = 0.8, standardize = TRUE)))
})

test_that("test new formula method", {
  data <- iris
  result1 <- classCV(data = data, target = "Species", models = "qda", train_params = list(split = 0.8, random_seed = 50))
  expect_no_error(result2 <- classCV(formula = Species ~ ., data = data, models = "qda", train_params = list(split = 0.8, random_seed = 50)))
  expect_equal(result1$metrics$lda$split,result2$metrics$lda$split)
})

test_that("k-fold CV no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "svm", train_params = list(n_folds = 3)))
})


test_that("k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "ann", size = 10,
                                    train_params = list(n_folds = 3, stratified = TRUE, random_seed = 50)))
})


test_that("train-test split and k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "naivebayes",
                                    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE)))
})

test_that("train-test split and k-fold CV without stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "multinom",
                                    train_params = list(split = 0.8, n_folds = 3)))
})


test_that("test imputation and missing data", {
  data <- iris
  # Introduce NA
  for(i in 1:ncol(data)){
    data[sample(1:nrow(data), size = round(nrow(data)*.10)),i] <- NA
  }
  
  # knn
  expect_warning(expect_warning(result <- classCV(data = data, target = "Species",
                                                  train_params = list(split = 0.8, n_folds = 4, stratified = TRUE),
                                                  impute_params = list(method = "knn_impute", args = list(neighbors = 5)),
                                                  models = "qda", model_params = list(final_model = TRUE))))
  
  # bag
  expect_warning(expect_warning(result <- classCV(data = data, target = "Species",
                                                  train_params = list(split = 0.8, n_folds = 4, stratified = FALSE),
                                                  impute_params = list(method = "bag_impute", args = list(trees = 5)),
                                                  models = "multinom", model_params = list(final_model = TRUE))))
  
  # complete cases only
  expect_warning(result <- classCV(data = data, target = "Species",
                                   train_params = list(split = 0.8, n_folds = 4, stratified = TRUE),
                                   models = "decisiontree", model_params = list(final_model = TRUE)))
})

test_that("test random seed", {
  
  data <- iris
  result_1 <- classCV(data = data, target = "Species",
                      train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50), models = "knn",
                      ks = 5)
  result_2 <- classCV(data = data, target = "Species",
                      train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50), models = "knn",
                      model_params = list(map_args = list(knn = list(ks = 5))))
  
  expect_equal(result_1$data_partitions$indices$split$train,result_2$data_partitions$indices$split$train)
  expect_equal(result_1$metrics$knn$cv,result_2$metrics$knn$cv)
  
  result_1 <- classCV(data = data, target = "Species",
                      train_params = list(split = 0.8, n_folds = 3, stratified = FALSE, random_seed = 50), models = "knn",
                      ks = 5)
  result_2 <- classCV(data = data, target = "Species",
                      train_params = list(split = 0.8, n_folds = 3, stratified = FALSE, random_seed = 50), models = "knn",
                      model_params = list(map_args = list(knn = list(ks = 5))))
  
  expect_equal(result_1$data_partitions$indices$split$train,result_2$data_partitions$indices$split$train)
  expect_equal(result_1$metrics$knn$cv,result_2$metrics$knn$cv)
})

test_that("running multiple models", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(booster = "gbtree", objective = "multi:softmax",
                                                                lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
                                                                max_depth = 6), nrounds = 10))
  
  
  
  expect_warning(result <- classCV(data = data, target = 5, models = c("knn", "svm", "gbm", "randomforest"), 
                                   train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50, remove_obs = TRUE),
                                   save = list(models = TRUE), model_params = list(map_args = args)))
  
  class(data$Species) <- "character"
  
  data$Species <- ifelse(data$Species == 1, 0, 1)
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(objective = "multi:softprob",num_class = 2,
                                                                eta = 0.3 ,max_depth = 6), nrounds = 3))
  
  expect_warning(result <- classCV(data = data, target = 5, models = c("knn", "svm", "logistic", "gbm", "randomforest"), 
                                    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50, remove_obs = TRUE),
                                    save = list(models = TRUE), model_params = list(map_args = args)))
})

test_that("n_cores", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "gbm" = list(params = list(booster = "gbtree", objective = "multi:softmax",
                                                                lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
                                                                max_depth = 6), nrounds = 10))
  
  expect_warning(result1 <- classCV(data = data, target = 5, models = c("knn", "svm", "gbm", "randomforest"), 
                                    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50),
                                    save = list(models = TRUE), model_params = list(map_args = args),
                                    parallel_configs = list(n_cores = 2, future.seed = 100)))
  
  expect_warning(result2 <- classCV(data = data, target = 5, models = c("knn", "svm", "gbm", "randomforest"), 
                                    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 50),
                                    save = list(models = TRUE), model_params = list(map_args = args),
                                    parallel_configs = list(n_cores = 2, future.seed = 100)))
  
  expect_equal(result1$metrics$knn$cv,result2$metrics$knn$cv)
  
})