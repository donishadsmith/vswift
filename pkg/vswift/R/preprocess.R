#Helper function for classCV and genFolds to check if inputs are valid
.error_handling <- function(data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = NULL, stratified = NULL,  random_seed = NULL,
                            impute_method = NULL, impute_args = NULL, mod_args = NULL, n_cores, call = NULL, ...){
  
  # List of valid inputs
  valid_inputs <- list(valid_models = c("lda","qda","logistic","svm","naivebayes","ann","knn","decisiontree",
                                      "randomforest", "multinom", "gbm"),
                       valid_imputes = c("simple", "missforest"))


  # Check if impute method is valid
  if(!is.null(impute_method)){
    if(!impute_method %in% valid_inputs[["valid_imputes"]]){
      stop("invalid impute method")
      }
    # Check if impute method is valid
    if(!is.null(impute_args)){
      if(all(impute_method == "missforest", class(impute_args) != "list")){
        stop("impute_args must be a list")
      }
    }
  }
  
  # Check if additional arguments are valid
  if(all(impute_method == "missforest",!is.null(impute_args))){
    vswift:::.check_additional_arguments(impute_method = impute_method, impute_args = impute_args, call = "imputation")
  }
  
  if(!is.null(mod_args)){
    if(class(mod_args) != "list"){
      stop("mod_args must be a list")
    }
    if(length(model_type) == 1){
      stop("mod_args used only when multiple models are specified")
    }
    if(!all(names(mod_args) %in% valid_inputs[["valid_models"]])){
      stop("invalid model in mod_args")
    }
  }
  
  if(!is.null(mod_args)){
    vswift:::.check_additional_arguments(model_type = model_type, mod_args = mod_args, call = "multiple")
  }
  
  if(length(list(...)) > 0){
    vswift:::.check_additional_arguments(model_type = model_type, call = "single", ...)
  }
  
  # Ensure fold size is valid
  if(any(n_folds %in% c(0,1), n_folds < 0, n_folds > 30,is.character(n_folds), n_folds != as.integer(n_folds))){
    stop("`n_folds` must be a non-negative integer from 3-30")
  }
  
  # Ensure split is between 0.5 to 0.9
  if(any(is.character(split), split < 0.5, split > 0.9)){
    stop("split must be a numeric value from between 0.5 and 0.9")
  }
  
  # Ensure valid target variable
  if(call == "classCV" || call == "genFolds" & stratified == TRUE){
    # Ensure target is also not in predictors 
    if(target %in% predictors){
      stop("target cannot also be a predictor")
    }
    
    # Ensure there is only one target variable
    if(length(target) != 1){
      stop("length of target must be 1")
    }
    
    # Check if target is in dataframe
    if(is.numeric(target)){
      if(!(target %in% c(1:ncol(data)))){
        stop("target not in dataframe")
      }
    } else if (is.character(target)){
      if(!(target %in% colnames(data))){
        stop("target not in dataframe")
      }
    } else {
      stop("target must be an integer or character")
    }
  }
  
  # Check if predictors are in data frame
  if(!is.null(predictors)){
    if(all(is.numeric(predictors))){
      check_x <- 1:dim(data)[1]
    } else if (all(is.character(predictors))){
      check_x <- colnames(data)[colnames(data) != target]
    } else {
      stop("predictors must be a character vector or integer vector")
    }
    if(!(all(predictors %in% check_x))){
      stop("at least one predictor is not in dataframe")
    }
  }
  
  #Ensure model_type has been assigned
  if(call == "classCV"){
    if(class(model_type) != "character"){
      stop("model_type must be a character or a vector containing characters")
    }
    if(!all(model_type %in% valid_inputs[["valid_models"]])){
      stop("invalid model in model_type")
    }
    if("logistic" %in% model_type & any(length(levels(factor(data[,target], exclude = NA))) != 2, !is.numeric(threshold), threshold < 0.30 || threshold > 0.70)){
      if(length(levels(factor(data[,target], exclude = NA))) != 2){
        stop("logistic regression requires a binary variable")
      } else {
        stop("threshold must a numeric value from 0.30 to 0.70")
      }
    }
  }
  # Check cores
  if(!is.null(n_cores)){
    if(all(!is.null(n_cores), is.null(n_folds))){
      stop("parallel processing only available if cross validation is specified")
    }
    if(!is.numeric(n_cores)){
      stop("number of cores must be a numeric value")
    }
    if(n_cores > detectCores()){
      stop(sprintf("more cores specified than available; only %s cores available but %s cores specified", detectCores(), n_cores))
    }
  }
}

# Helper function to turn character data into factors
.create_factor <- function(data = NULL, target = NULL, model_type = NULL){
  
  # Make target factor, could be factor, numeric, or character
  data[,target] <- factor(data[,target])
  # Check for columns that are characters and factors
  columns <- colnames(data)[as.vector(sapply(data,function(x) is.character(x)))]
  columns <- c(columns, colnames(data)[as.vector(sapply(data,function(x) is.factor(x)))])
  # Create list to store levels for svm model 
  data_levels <- list()
  # Turn character columns into factor
  for(col in columns){
    data[,col] <- factor(data[,col])
    if("svm" %in% model_type){
      data_levels[[col]] <- levels(data[,col])
    }
  }
  
  # Return output
  return(create_factor_output <- list("data" = data, "data_levels" = data_levels))
}


# Helper function for classCV to check if additional arguments are valid
.check_additional_arguments <- function(model_type = NULL, impute_method = NULL, impute_args = NULL, mod_args = NULL, call = NULL, ...){
  
  # Helper function to generate error message
  error_message <- function(method_name, invalid_args) {
    sprintf("The following arguments are invalid for %s or are incompatible with classCV: %s",
            method_name, paste(invalid_args, collapse = ","))
  }
  
  # List of valid arguments for each model type
  valid_args_list <- list(
    "lda" = c("prior", "method", "nu"),
    "qda" = c("prior", "method", "nu"),
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
  
  # Obtain user-specified models based on the number of models called
  if(call == "single"){
    methods <- model_type
    additional_args <- names(list(...))
  } else if(call == "multiple"){
    methods <- names(list(...))
  } else {
    methods <- impute_method
    additional_args <- names(impute_args)
  }
  
  # Obtain user-specified model arguments based on the number of models called
  for(method in methods){
    if(call == "single"){
      additional_args <- names(list(...))
    } else if(call == "multiple"){
      additional_args <- names(mod_args[[method]])
    } else {
      additional_args <- names(impute_args)
    }
    
    valid_args <- valid_args_list[[method]]
    invalid_args <- additional_args[which(!additional_args %in% valid_args)]
    
    if(length(invalid_args) > 0) {
      stop(error_message(method, invalid_args))
    }
  }
  
}

# Helper function for imputation
.imputation <- function(data = NULL, impute_method = NULL, impute_args = NULL){
  # Get data with missing columns
  missing_columns <- unique(data.frame(which(is.na(data), arr.ind = T))$col)
  
  # Check if there is missing data
  if(all(length(missing_columns) == 0,!is.null(impute_method))){
    warning("no missing data detected")
  } else if(all(length(missing_columns) > 0, !is.null(impute_method))){
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
  }
  
  # Warning for missing data if no imputation method selected or imputation fails to fill in some missing data
  miss <- nrow(data) - nrow(data[complete.cases(data),])
  if(miss  > 0){
    data <- data[complete.cases(data),]
    warning(sprintf("dataset contains %s observations with incomplete data only complete observations will be used"
                    ,miss))
  }
  
  if(exists("missing_information")){
    imputation_output <- list("preprocessed_data" = data, "imputation_information" = missing_information)
  } else {
    imputation_output <- list("preprocessed_data" = data)
  }
  
  # Return output
  return(imputation_output)
}


