#Helper function for classCV to remove unobserved data
.remove_obs <- function(training_data, test_data, target, fold = NULL){
  # Create empty list
  check_predictor_levels <- list()
  # Iterate over columns and check if column is a character or factor
  for(col in colnames(training_data[colnames(training_data) != target])){
    if(is.character(training_data[,col]) | is.factor(training_data[,col])){
      
      check_predictor_levels[[col]] <- names(table(training_data[,col]))[which(as.numeric(table(training_data[,col])) != 0)]
    }
  }
  
  #Check new columns and set certain predictors in NA if the model has not been trained on
  for(col in colnames(test_data)[colnames(test_data) != target]){
    if(is.character(test_data[,col]) | is.factor(test_data[,col])){
      missing <- names(table(test_data[,col]))[which(!(names(table(test_data[,col])) %in% check_predictor_levels[[col]]))]
      if(length(missing) > 0){
        delete_rows <- which(test_data[,col] %in% missing)
        observations <- row.names(test_data)[delete_rows]
        if(is.null(fold)){
          warning(sprintf("for predictor `%s` in test set has at least one class the model has not trained on\n these observations have been removed: %s", col,paste(observations, collapse = ",")))
        } else{
          warning(sprintf("for predictor `%s` in validation set - fold %s has at least one class the model has not trained on\n these observations have been removed: %s", col,fold, paste(observations, collapse = ",")))
        }
        test_data <- test_data[-delete_rows,] 
      }
    }
  }
  return(test_data)
}

# Helper function to calculate metrics

.calculate_metrics <- function(class = NULL, target = NULL, prediction_vector =  NULL, prediction_data = NULL){
  # Sum of true positives
  true_pos <- sum(prediction_data[,target][which(prediction_data[,target] == class)] == prediction_vector[which(prediction_data[,target] == class)])
  # Sum of false negatives
  false_neg <- sum(prediction_data[, target] == class & prediction_vector != class)
  # Sum of the false positive
  false_pos <- sum(prediction_vector == class) - true_pos
  # Calculate metrics 
  calculate_metrics_list <- list("precision" = true_pos/(true_pos + false_pos),
                                 "recall" = true_pos/(true_pos + false_neg))
  calculate_metrics_list[["f1"]] <- 2*(calculate_metrics_list[["precision"]]*calculate_metrics_list[["recall"]])/(calculate_metrics_list[["precision"]]+calculate_metrics_list[["recall"]])
  # Return list
  return(calculate_metrics_list)
}

# Helper function for classCV to create model
.generate_model <- function(model_type = NULL, formula = NULL, predictors = NULL, target = NULL, model_data = NULL, ...){
  switch(model_type,
         # Use double colon to avoid cluttering user space
         "lda" = {model <- MASS::lda(formula, data = model_data, ...)},
         "qda" = {model <- MASS::qda(formula, data = model_data, ...)},
         "logistic" = {model <- glm(formula, data = model_data , family = "binomial", ...)},
         "svm" = {model <- e1071::svm(formula, data = model_data, ...)},
         "naivebayes" = {model <- naivebayes::naive_bayes(formula = formula, data = model_data, ...)},
         "ann" = {model <- nnet::nnet(formula = formula, data = model_data, ...)},
         "knn" = {model <- kknn::train.kknn(formula = formula, data = model_data, ...)},
         "decisiontree" = {model <- rpart::rpart(formula = formula, data = model_data, ...)},
         "randomforest" = {model <- randomForest::randomForest(formula = formula, data = model_data, ...)},
         "multinom" = {model <- nnet::multinom(formula = formula, data = model_data, ...)},
         "gbm" = {
           model_data <- data.matrix(model_data)
           if(!is.null(predictors)){
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,predictors], label = model_data[,target])
           } else {
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,colnames(model_data)[colnames(model_data) != target]], label = model_data[,target])
           }
           model <- xgboost::xgb.train(data = xgb_data, ...)}
  )
  return(model)
}

# Helper function for classCV to predict 
.prediction <- function(model_type = NULL, model = NULL, prediction_data = NULL, predictors = NULL, class_dict = NULL, target = NULL){
  switch(model_type,
         "svm" = {prediction_vector <- predict(model, newdata = prediction_data)},
         "logistic" = {
           prediction_vector <- predict(model, newdata  = prediction_data, type = "response")
           prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
         "naivebayes" = {
           if(!is.null(predictors)){
             prediction_data <- prediction_data[,c(predictors,target)]
           } else {
             prediction_data <- prediction_data[,colnames(prediction_data)[colnames(prediction_data) != target]]
           }
           prediction_vector <- predict(model, newdata = prediction_data)},
         "ann" = {prediction_vector <- predict(model, newdata = prediction_data, type = "class")},
         "knn" = {prediction_vector <- predict(model, newdata = prediction_data)},
         "decisiontree" = {
           prediction_matrix <- predict(model, newdata = prediction_data)
           prediction_vector <- colnames(prediction_matrix)[apply(prediction_matrix, 1, which.max)]
         },
         "randomforest" = {prediction_vector <- predict(model, newdata = prediction_data)},
         "multinom" = {prediction_vector <- predict(model, newdata = prediction_data)},
         "gbm" = {
           prediction_data <- data.matrix(prediction_data)
           if(!is.null(predictors)){
             xgb_data <- xgboost::xgb.DMatrix(data = prediction_data[,predictors], label = prediction_data[,target])
           } else {
             xgb_data <- xgboost::xgb.DMatrix(data = prediction_data[,colnames(prediction_data)[colnames(prediction_data) != target]], label = prediction_data[,target])
           }
           predictions <- predict(model, xgb_data)
           prediction_matrix <- matrix(predictions, ncol = length(names(class_dict)), byrow = TRUE)
           
           prediction_vector <- apply(prediction_matrix, 1, which.max) - 1
         },
         prediction_vector <- predict(model, newdata = prediction_data)$class
  )
  
  return(prediction_vector)
}