# Testing classCV function
library(vswift)
library(testthat)

test_that("Fail due to `train_params` not being list", {
  data <- iris
  expect_error(result <- classCV(
    data = data, target = "Species", models = "lda",
    train_params = NULL
  ), "`train_params` must be a list")
})

test_that("Fail due to lack of nesting for `train_params`", {
  data <- iris
  expect_error(
    result <- classCV(
      data = data, target = "Species", models = "lda",
      train_params = list()
    ),
    "`train_params` must be a nested list containing one of the following valid keys: 'split', 'n_folds', 'stratified', 'random_seed', 'standardize', 'remove_obs'"
  )
})

test_that("test train-test split and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(
    data = data, target = "Species", models = "lda",
    train_params = list(split = 0.8, standardize = TRUE)
  ))
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result$metrics$lda$split
  expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
})

test_that("test train-test split and no stratified sampling w/ invalid key", {
  data <- iris
  expect_warning(result <- classCV(
    data = data, target = "Species", models = "lda",
    train_params = list(split = 0.8, standardize = TRUE, invalid_key = "1")
  ))
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result$metrics$lda$split
  expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
})

test_that("test new formula method", {
  data <- iris
  result1 <- classCV(
    data = data, target = "Species", models = "qda",
    train_params = list(split = 0.8, random_seed = 123)
  )
  expect_no_error(result2 <- classCV(
    formula = Species ~ ., data = data, models = "qda",
    train_params = list(split = 0.8, random_seed = 123)
  ))
  expect_equal(result1$metrics$lda$split, result2$metrics$lda$split)
  # Ensure values are greater than or equal to 0 and less than or equal to one
  split_df <- result1$metrics$qda$split
  expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
})

test_that("CV no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", models = "svm", train_params = list(n_folds = 3)))
  # Ensure values are greater than or equal to 0 and less than or equal to one
  cv_df <- result$metrics$svm$cv
  expect_true(all(cv_df[, 2:ncol(cv_df)] >= 0 & cv_df[, 2:ncol(cv_df)] <= 1))
})

test_that("CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(
    data = data, target = "Species", models = "nnet", size = 5,
    train_params = list(n_folds = 3, stratified = TRUE, random_seed = 123)
  ))
  expect_true(all(c("proportions", "indices") %in% names(result$class_summary)))
})

test_that("train-test split and k-fold CV with stratified", {
  data <- iris
  expect_no_error(result <- classCV(
    data = data, target = "Species", models = "naivebayes",
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE),
    save = list(data = TRUE)
  ))

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
      if (i != j) {
        expect_false(any(result$data_partitions$indices$cv[[i]] %in% result$data_partitions$indices$cv[[j]]))
      }
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
  expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
  expect_true(all(cv_df[, 2:ncol(cv_df)] >= 0 & cv_df[, 2:ncol(cv_df)] <= 1))
})

test_that("train-test split and k-fold CV without stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(
    data = data, target = "Species", models = "multinom",
    train_params = list(split = 0.8, n_folds = 3), save = list(data = TRUE)
  ))

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
      if (i != j) {
        expect_false(any(result$data_partitions$indices$cv[[i]] %in% result$data_partitions$indices$cv[[j]]))
      }
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
  expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
  expect_true(all(cv_df[, 2:ncol(cv_df)] >= 0 & cv_df[, 2:ncol(cv_df)] <= 1))
})

test_that("test final", {
  data <- iris
  expect_no_error(result <- classCV(
    data = data, target = "Species", models = "multinom",
    train_params = list(standardize = TRUE),
    model_params = list(final_model = TRUE)
  ))

  expect_true(all(!is.na(result$models$multinom$final)))

  # Should stop
  expect_error(result <- classCV(
    data = data, target = "Species", models = "multinom",
    train_params = list(standardize = TRUE),
    model_params = list(final_model = FALSE)
  ))
})

test_that("test final w imputation", {
  # Introduce NA
  data <- iris

  for (i in 1:ncol(data)) {
    data[sample(1:nrow(data), size = round(nrow(data) * .10)), i] <- NA
  }

  # Without folds
  expect_warning(expect_warning(result <- classCV(
    data = data, target = "Species", models = "multinom",
    train_params = list(standardize = TRUE),
    model_params = list(final_model = TRUE),
    impute_params = list(method = "impute_knn", args = list("neighbors" = 3)),
    save = list(data = TRUE)
  )))

  expect_true(all(!is.na(result$data_partitions$dataframes$final)))

  # With folds
  expect_warning(expect_warning(result <- classCV(
    data = data, target = "Species", models = "multinom",
    train_params = list(n_folds = 3),
    model_params = list(final_model = TRUE),
    impute_params = list(method = "impute_knn", args = list("neighbors" = 3)),
    save = list(data = TRUE)
  )))

  expect_true(all(!is.na(result$data_partitions$dataframes$final)))
})

test_that("test imputation and missing data", {
  data <- iris
  # Introduce NA
  for (i in 1:ncol(data)) {
    data[sample(1:nrow(data), size = round(nrow(data) * .10)), i] <- NA
  }

  # Add row with all missing features
  data[10, colnames(data)[colnames(data) != "Species"]] <- NA

  # knn
  expect_warning(expect_warning(result <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 4, stratified = TRUE),
    impute_params = list(method = "impute_knn", args = list(neighbors = 5)),
    models = "qda", model_params = list(final_model = TRUE)
  )))

  # bag
  expect_warning(expect_warning(result <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 4, stratified = FALSE),
    impute_params = list(method = "impute_bag", args = list(trees = 5)),
    models = "multinom", model_params = list(final_model = TRUE)
  )))

  # complete cases only
  expect_warning(result <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 4, stratified = TRUE),
    models = "decisiontree", model_params = list(final_model = TRUE),
    save = list(data = TRUE)
  ))
})

test_that("test random seed", {
  data <- iris
  result_1 <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123), models = "knn",
    ks = 5
  )
  result_2 <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123), models = "knn",
    model_params = list(map_args = list(knn = list(ks = 5)))
  )

  expect_equal(result_1$data_partitions$indices$split$train, result_2$data_partitions$indices$split$train)
  expect_equal(result_1$metrics$knn$cv, result_2$metrics$knn$cv)

  result_1 <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 3, stratified = FALSE, random_seed = 123), models = "knn",
    ks = 5
  )
  result_2 <- classCV(
    data = data, target = "Species",
    train_params = list(split = 0.8, n_folds = 3, stratified = FALSE, random_seed = 123), models = "knn",
    model_params = list(map_args = list(knn = list(ks = 5)))
  )

  expect_equal(result_1$data_partitions$indices$split$train, result_2$data_partitions$indices$split$train)
  expect_equal(result_1$metrics$knn$cv, result_2$metrics$knn$cv)
})

test_that("running multiple models", {
  data <- iris

  args <- list("knn" = list(ks = 3), "xgboost" = list(params = list(
    booster = "gbtree", objective = "multi:softmax",
    lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
    max_depth = 6
  ), nrounds = 10))

  expect_warning(result <- classCV(
    data = data, target = 5, models = c("knn", "svm", "xgboost", "randomforest"),
    train_params = list(
      split = 0.8, n_folds = 3, stratified = TRUE,
      random_seed = 123, remove_obs = TRUE
    ),
    save = list(models = TRUE), model_params = list(map_args = args)
  ))

  class(data$Species) <- "character"

  data$Species <- ifelse(data$Species == 1, 0, 1)

  args <- list(
    "knn" = list(ks = 3),
    "xgboost" = list(params = list(
      booster = "gbtree", objective = "multi:softmax",
      lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
      max_depth = 6
    ), nrounds = 10),
    "logistic" = list(maxit = 10000)
  )

  models <- c("knn", "svm", "logistic", "xgboost", "randomforest")
  expect_warning(expect_warning(result <- classCV(
    data = data, target = 5, models = models,
    train_params = list(
      split = 0.8, n_folds = 3, standardize = TRUE,
      stratified = TRUE, random_seed = 123, remove_obs = TRUE
    ),
    save = list(models = TRUE), model_params = list(map_args = args)
  )))


  # Ensure values are greater than or equal to 0 and less than or equal to one
  for (model in models) {
    split_df <- result$metrics[[model]]$split
    cv_df <- result$metrics[[model]]$cv
    expect_true(all(split_df[, 2:ncol(split_df)] >= 0 & split_df[, 2:ncol(split_df)] <= 1))
    expect_true(all(cv_df[, 2:ncol(cv_df)] >= 0 & cv_df[, 2:ncol(cv_df)] <= 1))
  }
})

test_that("n_cores", {
  data <- iris

  args <- list("knn" = list(ks = 3), "xgboost" = list(params = list(
    booster = "gbtree", objective = "multi:softmax",
    lambda = 0.0003, alpha = 0.0003, num_class = 3, eta = 0.8,
    max_depth = 6
  ), nrounds = 10))

  expect_warning(result1 <- classCV(
    data = data, target = 5, models = c("knn", "svm", "xgboost", "randomforest"),
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
    save = list(models = TRUE), model_params = list(map_args = args),
    parallel_configs = list(n_cores = 2, future.seed = 100)
  ))

  expect_warning(result2 <- classCV(
    data = data, target = 5, models = c("knn", "svm", "xgboost", "randomforest"),
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
    save = list(models = TRUE), model_params = list(map_args = args),
    parallel_configs = list(n_cores = 2, future.seed = 100)
  ))

  expect_equal(result1$metrics$knn$split, result2$metrics$knn$split)
  expect_equal(result1$metrics$knn$cv, result2$metrics$knn$cv)
})

test_that("ensure parallel and nonparallel outputs are equal", {
  data <- iris

  expect_no_error(result1 <- classCV(
    data = data, target = 5, models = "lda",
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
    save = list(models = TRUE)
  ))

  expect_no_error(result2 <- classCV(
    data = data, target = 5, models = "lda",
    train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
    save = list(models = TRUE),
    parallel_configs = list(n_cores = 2)
  ))

  expect_equal(result1$metrics$lda$split, result2$metrics$lda$split)
  expect_equal(result1$metrics$lda$cv, result2$metrics$lda$cv)
})

test_that("xgboost objectives-single", {
  df <- iris
  df$Species <- ifelse(df$Species == df$Species[1], 1, 0)

  bin_obj <- c("reg:logistic", "binary:logistic", "binary:hinge", "binary:logitraw")

  for (obj in bin_obj) {
    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = "xgboost",
      train_params = list(n_folds = 5, random_seed = 123),
      params = list(
        objective = obj,
        eta = 0.3,
        max_depth = 6
      ),
      nrounds = 10,
      save = list(models = T)
    )
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
  }

  multi_obj <- c("multi:softprob", "multi:softmax")

  for (obj in multi_obj) {
    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = "xgboost",
      train_params = list(n_folds = 5, random_seed = 123),
      params = list(
        objective = obj,
        num_class = 2,
        eta = 0.3,
        max_depth = 6
      ),
      nrounds = 10,
      save = list(models = T)
    )
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
  }
})

test_that("xgboost objectives-multi", {
  df <- iris
  df$Species <- ifelse(df$Species == df$Species[1], 1, 0)

  bin_obj <- c("reg:logistic", "binary:logistic", "binary:logitraw", "binary:hinge")
  multi_obj <- c("multi:softprob", "multi:softmax")

  for (obj in bin_obj) {
    args <- list("knn" = list(ks = 2), "xgboost" = list(params = list(
      booster = "gbtree", objective = obj,
      lambda = 0.0003, alpha = 0.0003, eta = 0.8,
      max_depth = 6
    ), nrounds = 10))

    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = c("xgboost", "knn"),
      train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
      model_params = list(map_args = args)
    )

    expect_true(all(!is.na(result$metrics$xgboost$split)))
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
    expect_true(all(!is.na(result$metrics$knn$split)))
    expect_true(all(!is.na(result$metrics$knn$cv)))
  }

  for (obj in multi_obj) {
    args <- list("knn" = list(ks = 2), "xgboost" = list(params = list(
      booster = "gbtree", objective = obj,
      lambda = 0.0003, alpha = 0.0003, num_class = 2,
      eta = 0.8, max_depth = 6
    ), nrounds = 10))

    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = c("xgboost", "knn"),
      train_params = list(split = 0.8, n_folds = 3, stratified = TRUE, random_seed = 123),
      model_params = list(map_args = args)
    )

    expect_true(all(!is.na(result$metrics$xgboost$split)))
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
    expect_true(all(!is.na(result$metrics$knn$split)))
    expect_true(all(!is.na(result$metrics$knn$cv)))
  }
})

test_that("binary target", {
  df <- iris
  df$Species <- ifelse(df$Species == df$Species[1], 1, 0)

  bin_obj <- c("reg:logistic", "binary:logistic", "binary:logitraw", "binary:hinge")

  for (obj in bin_obj) {
    args <- list("xgboost" = list(params = list(
      booster = "gbtree", objective = obj,
      lambda = 0.0003, alpha = 0.0003, eta = 0.8,
      max_depth = 6
    ), nrounds = 10))

    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = c("logistic", "xgboost"),
      train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
      model_params = list(map_args = args)
    )

    expect_true(all(!is.na(result$metrics$xgboost$split)))
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
    expect_true(all(!is.na(result$metrics$logistic$split)))
    expect_true(all(!is.na(result$metrics$logistic$cv)))

    result <- classCV(
      data = df,
      target = "Species",
      models = c("xgboost", "logistic"),
      train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
      model_params = list(map_args = args)
    )

    expect_true(all(!is.na(result$metrics$xgboost$split)))
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
    expect_true(all(!is.na(result$metrics$logistic$split)))
    expect_true(all(!is.na(result$metrics$logistic$cv)))
  }
})

test_that("test regularized", {
  df <- iris

  df$Species <- ifelse(df$Species == "setosa", "setosa", "not setosa")

  map_args <- list(regularized_logistic = list(alpha = 1, nfolds = 3))

  result <- classCV(
    data = df,
    target = "Species",
    models = c("regularized_logistic", "regularized_multinomial"),
    train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
    model_params = list(map_args = map_args)
  )

  expect_true(all(!is.na(result$metrics$regularized_logistic$split)))
  expect_true(all(!is.na(result$metrics$regularized_multinomial$split)))

  expect_true(all(!is.na(result$metrics$regularized_logistic$cv)))
  expect_true(all(!is.na(result$metrics$regularized_multinomial$cv)))


  # With final
  result <- classCV(
    data = df,
    target = "Species",
    models = c("regularized_logistic", "regularized_multinomial"),
    train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
    model_params = list(map_args = map_args, final_model = TRUE)
  )
})

test_that("test threshold no xgboost", {
  df <- iris

  df$Species <- ifelse(df$Species == "setosa", "setosa", "not setosa")

  mods <- names(vswift:::.MODEL_LIST)[!names(vswift:::.MODEL_LIST) == "xgboost"]
  map_args <- list(regularized_logistic = list(alpha = 1, nfolds = 3), knn = list(ks = 5), nnet = list(size = 4))

  result <- classCV(
    data = df,
    target = "Species",
    models = mods,
    train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
    model_params = list(map_args = map_args)
  )

  for (mod in mods) {
    expect_true(all(!is.na(result$metrics[[mod]]$split)))
    expect_true(all(!is.na(result$metrics[[mod]]$cv)))
  }
})

test_that("test threshold for xgboost", {
  df <- iris

  df$Species <- ifelse(df$Species == "setosa", "setosa", "not setosa")

  bin_obj <- c("reg:logistic", "binary:logistic", "binary:logitraw", "binary:hinge")

  for (obj in bin_obj) {
    args <- list("xgboost" = list(params = list(
      booster = "gbtree", objective = obj,
      lambda = 0.0003, alpha = 0.0003, eta = 0.8,
      max_depth = 6
    ), nrounds = 10))

    result <- classCV(
      data = df,
      formula = Species ~ .,
      models = "xgboost",
      train_params = list(split = 0.8, n_folds = 3, random_seed = 123),
      model_params = list(map_args = args)
    )

    expect_true(all(!is.na(result$metrics$xgboost$split)))
    expect_true(all(!is.na(result$metrics$xgboost$cv)))
  }
})
