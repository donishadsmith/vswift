#Helper function for classCV to remove unobserved data 
.remove_obs <- function(training_data, test_data, target, iter, method, preprocessed_data, output, stratified){
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
      delete_rows <- which(test_data[,col] %in% missing)
      observations <- row.names(test_data)[delete_rows]
      if(length(observations) > 0){
        warning(sprintf("for predictor `%s` in `%s` has at least one class the model has not trained on\n these observations have been removed: %s", col,iter,paste(observations, collapse = ",")))
        test_data <- test_data[-delete_rows,] 
        output[["sample_indices"]][[method]][[tolower(iter)]] <- as.numeric(row.names(test_data))
        # Update if stratified = TRUE
        if(stratified == TRUE){
          output[["sample_proportions"]][[method]][[tolower(iter)]] <- table(preprocessed_data[,target][output[["sample_indices"]][[method]][[tolower(iter)]]])/sum(table(preprocessed_data[,target][output[["sample_indices"]][[method]][[tolower(iter)]]])) 
        } 
      }
    }
  }
  remove_obs <- list("test_data" = test_data, "output" = output)
  return(remove_obs)
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
.generate_model <- function(model_type = NULL, formula = NULL, predictors = NULL, target = NULL, model_data = NULL, mod_args = mod_args, ...){
  # Turn to call
  if(!is.null(mod_args) & model_type != "gbm"){
    mod_args[[model_type]][["data"]] <- model_data
    mod_args[[model_type]][["formula"]] <- formula
  } 
  switch(model_type,
         # Use double colon to avoid cluttering user space
         "lda" = { 
           # Have to explicitly call the class "formula" methods
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(MASS:::lda.formula, mod_args[[model_type]])
           } else {
             model <- MASS::lda(formula, data = model_data, ...)
           }},
         "qda" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(MASS:::qda.formula, mod_args[[model_type]])
           } else {
             model <- MASS::qda(formula, data = model_data, ...)}},
         "logistic" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(glm, mod_args[[model_type]])
           } else {
             model <- glm(formula, data = model_data , family = "binomial", ...)}},
         "svm" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(e1071:::svm.formula, mod_args[[model_type]])
           } else {
             model <- e1071::svm(formula, data = model_data, ...)}},
         "naivebayes" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(naivebayes:::naive_bayes.formula, mod_args[[model_type]])
           } else {
             model <- naivebayes::naive_bayes(formula = formula, data = model_data, ...)}},
         "ann" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(nnet:::nnet.formula, mod_args[[model_type]])
           } else {
             model <- nnet::nnet(formula = formula, data = model_data, ...)}},
         "knn" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(kknn::train.kknn, mod_args[[model_type]])
           } else {
             model <- kknn::train.kknn(formula = formula, data = model_data, ...)}},
         "decisiontree" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(rpart::rpart, mod_args[[model_type]])
           } else {
             model <- rpart::rpart(formula = formula, data = model_data, ...)}},
         "randomforest" = {
           if(!is.null(mod_args[[model_type]])){
             model <- do.call(randomForest:::randomForest.formula, mod_args[[model_type]])
           } else {
             model <- randomForest::randomForest(formula = formula, data = model_data, ...)}},
         "multinom" = {
           if(!is.null(mod_args[[model_type]])){
           model <- do.call(nnet::multinom, mod_args[[model_type]])
         } else {
           model <- nnet::multinom(formula = formula, data = model_data, ...)}},
         "gbm" = {
           model_data <- data.matrix(model_data)
           if(!is.null(predictors)){
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,predictors], label = model_data[,target])
           } else {
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,colnames(model_data)[colnames(model_data) != target]], label = model_data[,target])
           }
           if(!is.null(mod_args[[model_type]])){
             mod_args[[model_type]][["data"]] <- xgb_data
             model <- do.call(xgboost::xgb.train, mod_args[[model_type]])
           } else {
             model <- xgboost::xgb.train(data = xgb_data, ...)}}
  )
  return(model)
}

# Helper function for classCV to predict 
.prediction <- function(model_type, model, prediction_data, threshold, predictors, class_dict, target){
  switch(model_type,
         "svm" = {prediction_vector <- predict(model, newdata = prediction_data)},
         "logistic" = {
           prediction_vector <- predict(model, newdata  = prediction_data, type = "response")
           prediction_vector <- ifelse(prediction_vector > threshold, 1, 0)},
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
