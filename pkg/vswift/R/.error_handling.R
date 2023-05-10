#Helper function for classCV and genFolds to check if inputs are valid
.error_handling <- function(data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = NULL, stratified = NULL,  random_seed = NULL,
                            impute_method = NULL, impute_args = NULL, call = NULL,...){
  #Valid models
  valid_models <- c("lda","qda","logistic","svm","naivebayes","ann","knn","decisiontree",
                    "randomforest", "multinom", "gbm")
  valid_impute <- c("simple", "missforest")
  # Ensure data is not NULL
  if(is.null(data)){
    stop("no input data")
  }
  
  # Check if impute method is valid
  if(!is.null(impute_method)){
    if(!impute_method %in% valid_impute){
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
    vswift:::.check_additional_arguments(impute_method = impute_method, impute_args = impute_args)
  }
  if(length(list(...)) > 0){
    vswift:::.check_additional_arguments(model_type = model_type, ...)
  }
  # Ensure n_folds is not an invalid number
  if(!is.data.frame(data)){
    stop("invalid input for data")
  }
  
  # Ensure fold size is valid
  if(any(n_folds %in% c(0,1), n_folds < 0, n_folds > 30,is.character(n_folds), n_folds != as.integer(n_folds))){
    stop("`n_folds` must be a non-negative integer from 3-30")
  }
  
  # Ensure split is between 0.5 to 0.9
  if(any(is.character(split), split < 0.5, split > 0.9)){
    stop("split must be a numeric value from between 0.5 and 0.9")
  }
  
  if(call == "classCV" || call == "genFolds" & stratified == TRUE){
    # Ensure target is not null
    if(is.null(target)){
      stop("target has no input")
    }
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
    }else if(is.character(target)){
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
    }else if(all(is.character(predictors))){
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
    if(any(is.null(model_type), !(model_type %in% valid_models))){
      stop(sprintf("%s is an invalid model_type", model_type))
    }
    if(model_type == "logistic" & any(length(levels(factor(data[,target], exclude = NA))) != 2, !is.numeric(threshold), threshold < 0.30 || threshold > 0.70)){
      if(length(levels(factor(data[,target], exclude = NA))) != 2){
        stop("logistic regression requires a binary variable")
      } else {
        stop("threshold must a numeric value from 0.30 to 0.70")
      }
    }
  }
}
