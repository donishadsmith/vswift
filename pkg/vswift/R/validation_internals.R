# Helper function for classCV that performs validation

.validation <- function(i, model_name, preprocessed_data, data_levels, formula, target, predictors, split, n_folds, mod_args, remove_obs, save_data, save_models, classCV_output , threshold, parallel, standardize, ...){
  #Create split word
  split_word <- unlist(strsplit(i, split = " "))
  if("Training" %in% split_word){
    # Assigning data split matrices to new variable 
    model_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["training"]], ]
    validation_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["test"]], ]
  } else if("Fold" %in% split_word){
    model_data <- preprocessed_data[-c(classCV_output[["sample_indices"]][["cv"]][[tolower(i)]]), ]
    validation_data <- preprocessed_data[c(classCV_output[["sample_indices"]][["cv"]][[tolower(i)]]), ]
  }
  
  if(all(class(standardize) %in% c("logical", "integer", "numeric"), standardize != FALSE)){
    model_data <- vswift:::.standardize(data = model_data, target = target, standardize = standardize)
    validation_data <- vswift:::.standardize(data = validation_data, target = target, standardize = standardize)
  }

  # Ensure columns have same levels
  if(model_name == "svm"){
    model_data[,names(data_levels)] <- data.frame(lapply(names(data_levels), function(col) factor(model_data[,col], levels = data_levels[[col]])))
    validation_data[,names(data_levels)] <- data.frame(lapply(names(data_levels), function(col) factor(validation_data[,col], levels = data_levels[[col]])))
  }
  # Generate model depending on chosen model_type
  if(any("Training" %in% split_word, "Fold" %in% split_word)){
    model <- vswift:::.generate_model(model_type = model_name, formula = formula, predictors = predictors, target = target, model_data = model_data, mod_args = mod_args, ...)
  }
  
  # Variable to select correct dataframe
  method <- ifelse("Fold" %in% split_word, "cv", "split")
  col <- ifelse(method == "cv", "Fold", "Set")

  # Create dataframe for validation testing
  if(remove_obs == TRUE){
    remove_obs_output <- vswift:::.remove_obs(training_data = model_data, test_data = validation_data, target = target, iter = i, method = method, preprocessed_data = preprocessed_data,
                                              output = classCV_output, stratified = stratified)
    validation_data <- remove_obs_output[["test_data"]]
    classCV_output <- remove_obs_output[["output"]]
  }
  
  # Create prediction data
  
  prediction_data <- list()
  
  if(i == "Training"){
    prediction_data[["Training"]] <- model_data
    prediction_data[["Test"]] <- validation_data
  } 
  else {
    prediction_data[[i]] <- validation_data
  }

  
  # Create variables used in for loops to calculate precision, recall, and f1
  if(model_name %in% c("logistic","gbm")){
    classes <- as.numeric(unlist(classCV_output[["class_dictionary"]]))
    class_names <- names(classCV_output[["class_dictionary"]])
  } else {
    class_names <- classes <- classCV_output[["classes"]][[target]]
  }
  
  # Save data
  if(save_data == TRUE){
    if(!is.data.frame(classCV_output[["saved_data"]][[method]][[tolower(i)]])){
      if(i == "Training"){
        #classCV_output[["saved_data"]][[method]][["training"]] <- preprocessed_data[classCV_output[["sample_indices"]][[method]][["training"]],]
        #classCV_output[["saved_data"]][[method]][["test"]] <- preprocessed_data[classCV_output[["sample_indices"]][[method]][["test"]],]
        classCV_output[["saved_data"]][[method]][["training"]] <- model_data
        classCV_output[["saved_data"]][[method]][["test"]] <- validation_data
      } else {
        #classCV_output[["saved_data"]][[method]][[tolower(i)]] <- preprocessed_data[classCV_output[["sample_indices"]][[method]][[tolower(i)]],
        classCV_output[["saved_data"]][[method]][[tolower(i)]] <- model_data
      }
    }
  }
  
  # Save model
  if(save_models == TRUE) classCV_output[["saved_models"]][[model_name]][[method]][[tolower(i)]] <- model
  
  
  # Get prediction
  if(i == "Training"){
    prediction_vector <- vswift:::.prediction(model_type = model_name, model = model, prediction_data = prediction_data, predictors = predictors, target = target, threshold = threshold, class_dict = classCV_output[["class_dictionary"]], var_names = c("Training","Test"))
  } else{
    prediction_vector <- vswift:::.prediction(model_type = model_name, model = model, prediction_data = prediction_data, predictors = predictors, target = target, threshold = threshold, class_dict = classCV_output[["class_dictionary"]], var_names = i)
  }
  for(j in names(prediction_vector)){
    # Calculate classification accuracy
    classification_accuracy <- sum(prediction_data[[j]][,target] == prediction_vector[[j]])/length(prediction_data[[j]][,target])
    classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == j),"Classification Accuracy"] <- classification_accuracy
    # Class positions to get the name of the class in class_names
    class_position <- 1
    for(class in classes){
      metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector[[j]], prediction_data = prediction_data[[j]])
      # Add information to dataframes
      classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == j),sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
      classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == j),sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
      classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == j),sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
      class_position <- class_position + 1
      # Warning is a metric is NA
      if(any(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))){
        metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))]
        warning(sprintf("at least on metric could not be calculated for class %s - %s: %s",class,j,paste(metrics, collapse = ",")))
      }
    }
  }
  
  # Reset class position
  class_position <- 1
  
  return(classCV_output)
}

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
.prediction <- function(model_type, model, prediction_data, threshold, predictors, class_dict, target, var_names){
  # Initialize prediction_vector
  prediction_vector <- list()
  for(var in var_names){
    switch(model_type,
           "svm" = {prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])},
           "logistic" = {
             prediction_vector[[var]] <- predict(model, newdata  = prediction_data[[var]], type = "response")
             prediction_vector[[var]] <- ifelse(prediction_vector[[var]] > threshold, 1, 0)
             },
           "naivebayes" = {
             if(!is.null(predictors)){
               prediction_data[[var]] <- prediction_data[[var]][,c(predictors,target)]
             } else {
               prediction_data[[var]] <- prediction_data[[var]][,colnames(prediction_data[[var]])[colnames(prediction_data[[var]]) != target]]
             }
             prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])},
           "ann" = {prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]], type = "class")},
           "knn" = {prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])},
           "decisiontree" = {
             prediction_matrix <- predict(model, newdata = prediction_data[[var]])
             prediction_vector[[var]] <- colnames(prediction_matrix)[apply(prediction_matrix, 1, which.max)]
           },
           "randomforest" = {prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])},
           "multinom" = {prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])},
           "gbm" = {
             prediction_data[[var]] <- data.matrix(prediction_data[[var]])
             if(!is.null(predictors)){
               xgb_data <- xgboost::xgb.DMatrix(data = prediction_data[[var]][,predictors], label = prediction_data[[var]][,target])
             } else {
               xgb_data <- xgboost::xgb.DMatrix(data = prediction_data[[var]][,colnames(prediction_data[[var]])[colnames(prediction_data[[var]]) != target]], label = prediction_data[[var]][,target])
             }
             predictions <- predict(model, xgb_data)
             prediction_matrix <- matrix(predictions, ncol = length(names(class_dict)), byrow = TRUE)
             prediction_vector[[var]] <- apply(prediction_matrix, 1, which.max) - 1
           },
           prediction_vector[[var]] <- predict(model, newdata = prediction_data[[var]])$class
    )
  }
  return(prediction_vector)
}

# Helper function to calculate metrics

.calculate_metrics <- function(class, target, prediction_vector, prediction_data){
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

.merge_list <- function(save_data, save_models, model_name, parallel_list, preprocessed_data, impute_method){
  classCV_output <- parallel_list[[1]]
  for(name in names(parallel_list)[-1]){
    # Merge metrics data
    if(save_data == TRUE){
      # Fix for parallel not storing test data
      if(!is.data.frame(classCV_output[["saved_data"]][["split"]][["test"]])){
        classCV_output[["saved_data"]][["split"]][["test"]] <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["test"]],]
      }
      if(all("Fold 1" %in% names(parallel_list), !is.data.frame(classCV_output[["saved_data"]][["cv"]][["fold 1"]]))){
        classCV_output[["saved_data"]][["cv"]][["fold 1"]] <- parallel_list[[name]][["saved_data"]][["cv"]][["fold 1"]]
      } else if(!is.data.frame(classCV_output[["saved_data"]][["cv"]][[name]])){
        classCV_output[["saved_data"]][["cv"]][[tolower(name)]] <- parallel_list[[name]][["saved_data"]][["cv"]][[tolower(name)]]
      }
    }
    if(save_models == TRUE){
      if(name == "Fold 1"){
        classCV_output[["saved_models"]][[model_name]][["cv"]] <- parallel_list[[name]][["saved_models"]][[model_name]][["cv"]]
      } else {
        classCV_output[["saved_models"]][[model_name]][["cv"]][[tolower(name)]] <- parallel_list[[name]][["saved_models"]][[model_name]][["cv"]][[tolower(name)]]
      }
    }
    if(is.na(classCV_output[["metrics"]][[model_name]][["cv"]][1,2])){
      classCV_output[["metrics"]][[model_name]][["cv"]] <- parallel_list[["Fold 1"]][["metrics"]][[model_name]][["cv"]]
    }
    else{
      dataframe_length <- ncol(classCV_output[["metrics"]][[model_name]][["cv"]])
      row <- as.numeric(unlist(strsplit(name, split = " "))[2])
      classCV_output[["metrics"]][[model_name]][["cv"]][row,2:dataframe_length] <- parallel_list[[name]][["metrics"]][[model_name]][["cv"]][row,2:dataframe_length]
    }
    
    if(!is.null(impute_method)){
      if(name == "Fold 1"){
        classCV_output[["imputation"]][["cv"]] <- parallel_list[[name]][["imputation"]][["cv"]]
      } else {
        classCV_output[["imputation"]][["cv"]][[tolower(name)]] <- parallel_list[[name]][["imputation"]][["cv"]][[tolower(name)]]
      }
    }

  }
  return(classCV_output)
}


