# Testing classCV function
library(vswift)
library(testthat)
# Test that each model works with train-test splitting alone
test_that("test train-test split and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "lda",
                                    train_params = list(split = 0.8, standardize = TRUE)))
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result$metrics$lda$split
  expect_true(all(split_df[,2:ncol(split_df)] >= 0 & split_df[,2:ncol(split_df)] <= 1))
})

test_that("test new formula method", {
  data <- iris
  result1 <- classCV(data = data, target = "Species", models = "qda", train_params = list(split = 0.8, random_seed = 50))
  expect_no_error(result2 <- classCV(formula = Species ~ ., data = data, models = "qda",
                                     train_params = list(split = 0.8, random_seed = 50)))
  expect_equal(result1$metrics$lda$split,result2$metrics$lda$split)
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result1$metrics$qda$split
  expect_true(all(split_df[,2:ncol(split_df)] >= 0 & split_df[,2:ncol(split_df)] <= 1))
})

test_that("k-fold CV no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "svm", train_params = list(n_folds = 3)))
  # Ensure values are greater than or equal to 0 and less than or equal to one
  cv_df <- result$metrics$svm$cv
  expect_true(all(cv_df[,2:ncol(cv_df)] >= 0 & cv_df[,2:ncol(cv_df)] <= 1))
})


test_that("k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "ann", size = 10,
                                    train_params = list(n_folds = 3, stratified = TRUE, random_seed = 50)))
})


test_that("train-test split and k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "naivebayes",
                                    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE),
                                    save = list(data = TRUE)))
  
  # Check that data partition indices between train test split are independent
  expect_false(any(result$data_partitions$split$train %in% result$data_partitions$split$train))
  
  # Check that indices are assigned correctly when dataframes are made for modeling
  expect_true(all(result$data_partitions$indices$split$train %in% rownames(result$data_partitions$dataframes$split$train)))
  expect_true(all(result$data_partitions$indices$split$test %in% rownames(result$data_partitions$dataframes$split$test)))
  
  train_len <- length(result$data_partitions$indices$split$train)
  test_len <- length(result$data_partitions$indices$split$test)
  expect_true(c(train_len + test_len) == nrow(data))
  
  folds <- paste0("fold", 1:3)
  
  # Check that data partition indices between folds are independent
  for (i in folds) {
    for (j in folds) {
      if (i != j)
        expect_false(any(result$data_partitions$indices$cv[[i]] %in% result$data_partitions$indices$cv[[j]]))
    }
  }
  
  # Check that fold train and test data are correct
  for (i in folds) {
    train_indxs <- as.numeric(unlist(result$data_partitions$indices$cv[!names(result$data_partitions$indices$cv) == i]))
    test_indxs <- as.numeric(unlist(result$data_partitions$indices$cv[[i]]))
    
    expect_true(all(train_indxs %in% rownames(result$data_partitions$dataframes$cv[[i]]$train)))
    expect_true(all(test_indxs %in% rownames(result$data_partitions$dataframes$cv[[i]]$test)))

  }
  
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result$metrics$naivebayes$split
  cv_df <- result$metrics$naivebayes$cv
  expect_true(all(split_df[,2:ncol(split_df)] >= 0 & split_df[,2:ncol(split_df)] <= 1))
  expect_true(all(cv_df[,2:ncol(cv_df)] >= 0 & cv_df[,2:ncol(cv_df)] <= 1))
})


test_that("train-test split and k-fold CV without stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "multinom",
                                    train_params = list(split = 0.8, n_folds = 3), save = list(data = TRUE)))
  
  # Test again since regular split uses different code from stratified
  # Check that data partition indices between train test split are independent
  expect_false(any(result$data_partitions$split$train %in% result$data_partitions$split$train))
  
  # Check that indices are assigned correctly when dataframes are made for modeling
  expect_true(all(result$data_partitions$indices$split$train %in% rownames(result$data_partitions$dataframes$split$train)))
  expect_true(all(result$data_partitions$indices$split$test %in% rownames(result$data_partitions$dataframes$split$test)))
  
  train_len <- length(result$data_partitions$indices$split$train)
  test_len <- length(result$data_partitions$indices$split$test)
  expect_true(c(train_len + test_len) == nrow(data))
  
  folds <- paste0("fold", 1:3)
  
  # Check that data partition indices between folds are independent
  for (i in folds) {
    for (j in folds) {
      if (i != j)
        expect_false(any(result$data_partitions$indices$cv[[i]] %in% result$data_partitions$indices$cv[[j]]))
    }
  }
  
  # Check that fold train and test data are correct
  for (i in folds) {
    train_indxs <- as.numeric(unlist(result$data_partitions$indices$cv[!names(result$data_partitions$indices$cv) == i]))
    test_indxs <- as.numeric(unlist(result$data_partitions$indices$cv[[i]]))
    
    expect_true(all(train_indxs %in% rownames(result$data_partitions$dataframes$cv[[i]]$train)))
    expect_true(all(test_indxs %in% rownames(result$data_partitions$dataframes$cv[[i]]$test)))
  }
  
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result$metrics$multinom$split
  cv_df <- result$metrics$multinom$cv
  expect_true(all(split_df[,2:ncol(split_df)] >= 0 & split_df[,2:ncol(split_df)] <= 1))
  expect_true(all(cv_df[,2:ncol(cv_df)] >= 0 & cv_df[,2:ncol(cv_df)] <= 1))
})


test_that("test final", {
  data <- iris
  expect_warning(result <- classCV(data = data, target = "Species", models = "multinom",
                                    train_params = list(standardize = TRUE),
                                    model_params = list(final_model = TRUE)))
})

test_that("test final w imputation", {
  # Introduce NA
  data <- iris

  for (i in 1:ncol(data)) {
    data[sample(1:nrow(data), size = round(nrow(data)*.10)),i] <- NA
  }

  expect_warning(expect_warning(result <- classCV(data = data, target = "Species", models = "multinom",
                                   train_params = list(standardize = TRUE),
                                   model_params = list(final_model = TRUE),
                                   impute_params = list(method = "knn_impute", args = list("neighbors" = 3)),
                                   save = list(data = TRUE))))
  
  
  
  expect_true(all(!is.na(result$data_partitions$dataframes$final)))
})

test_that("test imputation and missing data", {
  data <- iris
  # Introduce NA
  for (i in 1:ncol(data)) {
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
                                   models = "decisiontree", model_params = list(final_model = TRUE),
                                   save = list(data = TRUE)))
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
                                   train_params = list(split = 0.8, n_folds = 3, stratified = TRUE,
                                                       random_seed = 50, remove_obs = TRUE),
                                   save = list(models = TRUE), model_params = list(map_args = args)))
  
  class(data$Species) <- "character"
  
  data$Species <- ifelse(data$Species == 1, 0, 1)
  
  args <- list("knn" = list(ks = 3),
               "gbm" = list(params = list(booster = "gbtree", objective = "multi:softmax",
                                          lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
                                          max_depth = 6), nrounds = 10),
               "logistic" = list(maxit = 10000))
  
  models = c("knn", "svm", "logistic", "gbm", "randomforest")
  expect_warning(expect_warning(result <- classCV(data = data, target = 5, models = models, 
                                    train_params = list(split = 0.8, n_folds = 3, standardize = TRUE,
                                                        stratified = TRUE, random_seed = 50, remove_obs = TRUE),
                                    save = list(models = TRUE), model_params = list(map_args = args))))
  
  
  # Ensure values are greater than or equal to 0 and less than or equal to one
  for (model in models) {
    split_df <- result$metrics[[model]]$split
    cv_df <- result$metrics[[model]]$cv
    expect_true(all(split_df[,2:ncol(split_df)] >= 0 & split_df[,2:ncol(split_df)] <= 1))
    expect_true(all(cv_df[,2:ncol(cv_df)] >= 0 & cv_df[,2:ncol(cv_df)] <= 1))
  }
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