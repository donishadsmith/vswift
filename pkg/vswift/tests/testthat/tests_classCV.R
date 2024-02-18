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

test_that("test new formula method for all models", {
  data <- iris
  expect_no_error(result <- classCV(formula = Species ~ ., data = data, split = 0.8, model_type = "lda"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "qda"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "svm"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "decisiontree"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "randomforest"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "knn", ks = 5))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "ann", size = 10))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "naivebayes"))
  expect_no_error(result <- classCV(formula = Species ~ .,data = data,  split = 0.8, model_type = "multinom"))
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
    expect_warning(result <- classCV(formula = diabetes ~ ., data = PimaIndiansDiabetes, split = 0.8, n_folds = 5, model_type = "logistic"))
    PimaIndiansDiabetes$diabetes <- as.character(PimaIndiansDiabetes$diabetes)
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "neg"), "diabetes"] <- 0
    PimaIndiansDiabetes[which(PimaIndiansDiabetes$diabetes == "pos"), "diabetes"] <- 1
    PimaIndiansDiabetes$diabetes <- as.numeric(PimaIndiansDiabetes$diabetes)
    expect_no_error(result <- classCV(formula = diabetes ~ ., data = PimaIndiansDiabetes, split = 0.8, model_type = "logistic"))
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
  for(i in 1:ncol(data)){
    data[sample(1:nrow(data), size = round(nrow(data)*.10)),i] <- NA
  }
  
  # knn
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "knn_impute", impute_args = list(neighbors = 5), model_type = "lda", stratified = TRUE))
  
  # bag
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, impute_method = "bag_impute", impute_args = list(trees = 5), model_type = "lda", stratified = TRUE))
  
  # complete cases only
  expect_warning(result <- classCV(data = data, target = "Species", split = 0.8, n_folds = 3, model_type = "lda", stratified = TRUE))
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
  
  data$Species <- ifelse(data$Species == "setosa","setosa","not setosa")
  
  args <- list("knn" = list(ks = 3), "ann" = list(size = 10), "gbm" = list(params = list(objective = "multi:softprob",num_class = 2,eta = 0.3,max_depth = 6), nrounds = 10))
  
  expect_no_error(result <- classCV(data = data, target = 5, split = 0.8, model_type = c("knn", "randomforest","ann","svm", "lda", "qda", "gbm", "logistic"), 
                                    n_folds = 5, mod_args = args, save_data = T, save_models = T, remove_obs = T, stratified = T))
})