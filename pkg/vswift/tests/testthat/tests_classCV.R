# Testing classCV function
library(vswift)
library(testthat)
# Test that each model works with train-test splitting alone
test_that("test train-test split for all models and no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "ann", size = 10))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "naivebayes"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "multinom"))
  count <- 0
  data$Species <- as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] <- as.character(count)
    count <- count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    # Create binary
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("test train-test split for all models with stratified sampling", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "ann", size = 10, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "naivebayes", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "multinom", stratified = TRUE))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    # Create binary
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("k-fold CV for all models no stratified sampling", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "ann", size = 10))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "naivebayes"))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "multinom"))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("k-fold CV for all models", {
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "ann", size = 10, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "naivebayes", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "multinom", stratified = TRUE))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", n_folds = 3, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("train-test split and k-fold CV for all models", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "qda", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "svm", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "decisiontree", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "randomforest", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", ks = 5, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "ann", size = 10, stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "naivebayes", stratified = TRUE))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "multinom", stratified = TRUE))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10, stratified = TRUE))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("train-test split and k-fold CV for all models without stratified sampling", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "qda"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "svm"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "decisiontree"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "randomforest"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "ann", size = 10))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "naivebayes"))
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "multinom"))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", split = 0.8, n_folds = 5, model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})


test_that("test imputation and missing data", {
  
  data <- iris
  # Introduce NA
  data <- missForest::prodNA(data)
  # randomforest
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE), model_type = "lda", stratified = TRUE))
  
  # simple
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "simple", model_type = "lda", stratified = TRUE))
  
  # complete cases only
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE))
})

test_that("test imputations", {
  if(requireNamespace("missForest", quietly = TRUE)){
    data <- iris
    # Introduce NA
    data <- missForest::prodNA(data)
    # randomforest no impute arguments
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", model_type = "lda", stratified = TRUE))
    # randomforest with impute arguments
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE, maxiter = 1000, maxnodes = 5), model_type = "lda", stratified = TRUE, final_model = TRUE))
    # random forest with incorrect impute arguments
    expect_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "missforest", impute_args = list(verbose = TRUE, maxiter = 1000, try = 5), model_type = "lda", stratified = TRUE, final_model = TRUE))
    # simple
    expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "simple", model_type = "lda", stratified = TRUE, final_model = TRUE))
    # complete cases only
    expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE, final_model = TRUE))

  } else {
    skip("missForest package not available")
  }})

test_that("test saving features", {
  data <- iris
  
  expect_no_error(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE,
                                    save_models = TRUE, save_data = TRUE))
  
})

test_that("test input errors", {
  
  data <- iris
  
  expect_error(result <- classCV(target = "Species", split = 0.1, model_type = "lda"))
  # split out of range
  expect_error(result <- classCV(data = data, target = "Species", split = 0.1, model_type = "lda"))
  # n_folds out of range
  expect_error(result <- classCV(data = data, target = "Species", n_folds = 31, model_type = "lda"))
  # model not specified
  expect_error(result <- classCV(data = data, target = "Species", n_folds = 31, model_type = "lda"))
  # no target
  expect_error(result <- classCV(data = data, n_folds = 31, model_type = "lda"))
  # target out of range
  expect_error(result <- classCV(data = data, target = 5, n_folds = 31, model_type = "lda"))
  # target also predictor
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, model_type = "lda"))
  # incorrect imputation
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, impute_method = "knn", model_type = "lda"))
  expect_error(result <- classCV(data = data, target = 5, predictors = 1:5, n_folds = 31, model_type = "cnn"))
})

test_that("test final models only", {
  
  data <- iris
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "lda", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "qda", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "svm", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "decisiontree", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "randomforest", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "knn", ks = 5, final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "ann", size = 10, final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "naivebayes", final_model = T))
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "multinom", final_model = T))
  count = 0
  data$Species = as.character(data$Species)
  for(x in names(table(data$Species))){
    data$Species[which(data$Species == x)] = as.character(count)
    count = count + 1
  }
  expect_no_error(result <- classCV(data = data, target = "Species", model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10, final_model = T))
  # Create binary
  if(requireNamespace("mlbench", quietly = TRUE)){
    data(PimaIndiansDiabetes, package = "mlbench")
    # Convert characters to zero and one; expect warning that this conversion is happening
    expect_warning(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes", final_model = TRUE, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(data = PimaIndiansDiabetes, target = "diabetes",  final_model = TRUE,  model_type = "logistic"))
  } else {
    skip("mlbench package not available")
  }
})

test_that("test random seed", {
  
  data <- iris
  result_1 <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", random_seed = 123, stratified = TRUE)
  result_2 <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", random_seed = 123, stratified = TRUE)
  
  expect_equal(result_1$sample_indices$split$training,result_2$sample_indices$split$training)
  expect_equal(result_1$metrics$cv,result_2$metrics$cv)
})

test_that("running multiple models", {
  data <- iris 
  
  args <- list("knn" = list(ks = 3), "ann" = list(size = 10))
  
  expect_no_error(result <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "randomforest","ann","svm"), 
                    n_folds = 5, mod_args = args, save_data = T, save_models = T, remove_obs = T, stratified = T))
})
