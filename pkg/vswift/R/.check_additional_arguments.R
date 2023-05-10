# Helper function for imputation

.impute <- function(data = NULL, missing_columns = NULL, impute_method = NULL, impute_args = NULL){
  # Create empty list to store information
  missing_information <- list()
  # Get missing column information
  for(col in missing_columns){
    missing_information[[names(data)[col]]][["missing"]] <- length(which(is.na(data[,col])))
  }
  # switch statement
  switch(impute_method,
         "simple" = {
           for(col in missing_columns){
             if(is.character(data[,col]) || is.factor(data[,col])){
             frequent_class <- names(which.max(table(data[,col])))
             missing_information[[names(data)[col]]][["mode"]] <- frequent_class
             data[which(is.na(data[,col])),col]  <- frequent_class
           }
             else{
             # Check distribution
             missing_information[[names(data)[col]]][["shapiro_p.value"]] <- shapiro_p.value <- shapiro.test(data[,col])[["p.value"]]
             # If less than 0.5, distribution is not normal median will be used
             if(shapiro_p.value < 0.05){
               missing_information[[names(data)[col]]][["median"]] <- data[which(is.na(data[,col])),col] <- median(data[,col], na.rm = TRUE)
             } else {
               missing_information[[names(data)[col]]][["mean"]] <- data[which(is.na(data[,col])),col] <- mean(data[,col], na.rm = TRUE)
             }
           }
         }},
         "missforest" = {
           if(!is.null(impute_args)){
             impute_args[["xmis"]] <- data
             missforest_output <- do.call(missForest::missForest, impute_args)
           } else {
             missforest_output <- missForest::missForest(data)
           }
           missing_information[["missForest"]] <- missforest_output
           data <- missforest_output[["ximp"]]
         })

  impute_output <- list("preprocessed_data" = data, "impute_info" = missing_information)
  return(impute_output)
}

# Helper function for classCV to check if additional arguments are valid
.check_additional_arguments <- function(model_type = NULL, impute_method = NULL, impute_args = NULL, ...){
  
  if(length(names(list(...))) > 0){
    method <- model_type
    additional_args <- names(list(...))
  } else {
    method <- impute_method
    additional_args <- names(impute_args)
  }
  # Helper function to generate error message
  error_message <- function(method_name, invalid_args) {
    sprintf("The following arguments are invalid for %s or are incompatible with classCV: %s",
            method_name, paste(invalid_args, collapse = ","))
  }
  
  # List of valid arguments for each model type
  valid_args_list <- list(
    "lda" = c("grouping", "prior", "method", "nu"),
    "qda" = c("grouping", "prior", "method", "nu"),
    "logistic" = c("weights", "start", "etastart", "mustart", "offset", "control", "contrasts", "intercept", "singular.ok", "type", "maxit"),
    "svm" = c("scale", "type", "kernel", "degree", "gamma", "coef0", "cost", "nu", "class.weights", "cachesize", "tolerance", "epsilon", "shrinking", "cross", "probability", "fitted"),
    "naivebayes" = c("prior", "laplace", "usekernel", "usepoisson"),
    "ann" = c("weights", "size", "Wts", "mask", "linout", "entropy", "softmax", "censored", "skip", "rang", "decay", "maxit", "Hess", "trace", "MaxNWts", "abstol", "reltol"),
    "knn" = c("kmax", "ks", "distance", "kernel", "scale", "contrasts", "ykernel"),
    "decisiontree" = c("weights", "method", "parms", "control", "cost"),
    "randomforest" = c("ntree", "mtry", "weights", "replace", "classwt", "cutoff", "strata", "nodesize", "maxnodes", "importance", "localImp", "nPerm", "proximity", "oob.prox", "norm.votes", "do.trace", "keep.forest", "corr.bias", "keep.inbag"),
    "multinom" = c("weights", "Hess"),
    "gbm" = c("params", "nrounds", "verbose", "print_every_n", "early_stopping_rounds"),
    "missforest" = c("maxiter","ntree","variablewise","decreasing","verbose","mtry", "replace", "classwt", "cutoff","strata", "sampsize", "nodesize", "maxnodes")
  )
  
  valid_args <- valid_args_list[[method]]
  invalid_args <- additional_args[which(!additional_args %in% valid_args)]
  
  if(length(invalid_args) > 0) {
    stop(error_message(method, invalid_args))
  }
}

#Helper function for classCV for stratified sampling
.stratified_sampling <- function(data, type, output, target, split = NULL, k = NULL,
                                 random_seed = NULL){
  switch(type,
         "split" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           
           #Get class indices
           class_indices <- output[["class_indices"]]
           #Split sizes
           training_n <- round(nrow(data)*split,0)
           test_n <- nrow(data) - training_n
           #Initialize list
           output[["sample_indices"]][["split"]] <- list()
           output[["sample_proportions"]][["split"]] <- list()
           # Extract indices for each class
           for(class in names(output[["class_proportions"]])){
             #Check if sampling possible
             vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = training_n)
             #Store indices for training set
             output[["sample_indices"]][["split"]][["training"]] <- c(output[["sample_indices"]][["split"]][["training"]] ,sample(class_indices[[class]],size = round(training_n*output[["class_proportions"]][[class]],0), replace = F))
             #Remove indices to not add to test set
             class_indices[[class]] <- class_indices[[class]][!(class_indices[[class]] %in% output[["sample_indices"]][["split"]][["training"]])]
             # Check if sampling possible
             vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = test_n)
             #Add indices for test set
             output[["sample_indices"]][["split"]][["test"]] <- c(output[["sample_indices"]][["split"]][["test"]] ,sample(class_indices[[class]],size = round(test_n*output[["class_proportions"]][[class]],0), replace = F))
           }
           #Store proportions of data in training set
           output[["sample_proportions"]][["split"]][["training"]] <- table(data[,target][output[["sample_indices"]][["split"]][["training"]]])/sum(table(data[,target][output[["sample_indices"]][["split"]][["training"]]]))
           #Store proportions of data  in test set
           output[["sample_proportions"]][["split"]][["test"]] <- table(data[,target][output[["sample_indices"]][["split"]][["test"]]])/sum(table(data[,target][output[["sample_indices"]][["split"]][["test"]]]))
           #Output
           stratified_sampling_output <- list("output" = output)
         },
         "k-fold" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           
           #Get class indices
           class_indices <- output[["class_indices"]]
           #Initialize sample_indices for cv since it will be three levels
           output[["sample_indices"]][["cv"]] <- list()
           
           # Create folds
           for(i in 1:k){
             #Keep initializing variable
             fold_idx <- c()
             output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
             #fold size; try to undershoot for excess
             fold_size <- floor(nrow(data)/k)
             # Assign class indices to each fold
             for(class in names(output[["class_proportions"]])){
               #Check if sampling possible
               vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = fold_size)
               #Check if sampling possible
               fold_idx <- c(fold_idx, sample(class_indices[[class]],size = floor(fold_size*output[["class_proportions"]][[class]]), replace = F))
               #Remove already selected indices
               class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
             }
             #Add indices to list
             output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
             #Update proportions
             output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
           }
           #Deal with excess indices
           excess <- nrow(data) - length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
           if(excess > 0){
             for(class in names(output[["class_proportions"]])){
               fold_idx <- class_indices[[class]]
               if(length(fold_idx) > 0){
                 leftover <- rep(1:k,length(fold_idx))[1:length(fold_idx)]
                 for(i in 1:length(leftover)){
                   #Add indices to list
                   output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
                   #Update class proportions
                   output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
                 }
               }
             }
           }
           #Output
           stratified_sampling_output <- list("output" = output)
         }
  )
}

# Helper function for .stratified_sampling to error check
.stratified_check <- function(class, class_indices, output, n){
  # Check if there are zero indices for a specific class
  if(round(n*output[["class_proportions"]][[class]],0) == 0){
    stop(sprintf("0 indices selected for %s class\n not enough samples for stratified sampling", class))
  }
  # Check if there there are enough indices in class for proper assignment
  if(round(n*output[["class_proportions"]][[class]],0) > length(class_indices[[class]])){
    stop(sprintf("not enough samples of %s class for stratified sampling", class))
  }
}

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

.calculate_metrics <- function(class, target, prediction_vector, model_data){
  # Sum of true positives
  true_pos <- sum(model_data[,target][which(model_data[,target] == class)] == prediction_vector[which(model_data[,target] == class)])
  # Sum of false negatives
  false_neg <- sum(model_data[, target] == class & prediction_vector != class)
  # Sum of the false positive
  false_pos <- sum(prediction_vector == class) - true_pos
  # Calculate metrics 
  calculate_metrics_list <- list("precision" = true_pos/(true_pos + false_pos),
                                 "recall" = true_pos/(true_pos + false_neg))
  calculate_metrics_list[["f1"]] <- 2*(calculate_metrics_list[["precision"]]*calculate_metrics_list[["recall"]])/(calculate_metrics_list[["precision"]]+calculate_metrics_list[["recall"]])
  # Return list
  return(calculate_metrics_list)
}

# Helper function for classCV to predict 

.prediction <- function(model_type = NULL, model = NULL, model_data = NULL, predictors = NULL, class_dict = NULL, target = NULL){
  switch(model_type,
         "svm" = {prediction_vector <- predict(model, newdata = model_data)},
         "logistic" = {
           prediction_vector <- predict(model, newdata  = model_data, type = "response")
           prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
         "naivebayes" = {
           if(!is.null(predictors)){
             model_data <- model_data[,c(predictors,target)]
           } else {
             model_data <- model_data[,colnames(model_data)[colnames(model_data) != target]]
           }
           prediction_vector <- predict(model, newdata = model_data)},
         "ann" = {prediction_vector <- predict(model, newdata = model_data, type = "class")},
         "knn" = {prediction_vector <- predict(model, newdata = model_data)},
         "decisiontree" = {
           prediction_matrix <- predict(model, newdata = model_data)
           prediction_vector <- colnames(prediction_matrix)[apply(prediction_matrix, 1, which.max)]
           },
         "randomforest" = {prediction_vector <- predict(model, newdata = model_data)},
         "multinom" = {prediction_vector <- predict(model, newdata = model_data)},
         "gbm" = {
           model_data <- as.matrix(model_data)
           if(!is.null(predictors)){
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,predictors], label = model_data[,target])
           } else {
             xgb_data <- xgboost::xgb.DMatrix(data = model_data[,colnames(model_data)[colnames(model_data) != target]], label = model_data[,target])
           }
           predictions <- predict(model, xgb_data)
           prediction_matrix <- matrix(predictions, ncol = length(names(class_dict)), byrow = TRUE)
           
           prediction_vector <- apply(prediction_matrix, 1, which.max) - 1
         },
         prediction_vector <- predict(model, newdata = model_data)$class
  )
  
  return(prediction_vector)
}

