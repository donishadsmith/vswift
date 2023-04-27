#Helper function for categorical_cv_split and stratified_split to check if inputs are valid
.error_handling <- function(data = NULL, y_col = NULL,x_col = NULL,fold_n = NULL,split = NULL, model_type = NULL, stratified = NULL,  random_seed = NULL,
                            call = NULL,...){
  #Valid models
  valid_models <- c("lda","qda","logistic","svm","naivebayes","ann","knn","decisiontree",
                    "randomforest")
  if(all(!is.null(random_seed),!is.numeric(random_seed))){
    stop("random_seed must be a numerical scalar value")
  }
  # Ensure fold_n is not an invalid number
  if(!is.data.frame(data)){
    stop("invalid input for data")
  }
  if(any(fold_n %in% c(0,1), fold_n < 0, fold_n > 30,is.character(fold_n), fold_n != as.integer(fold_n))){
    stop(sprintf("fold_n = %s is not a valid input. `fold_n` must be a non-negative integer between 2-30",fold_n))
  }
  # Ensure split is between 0.5 to 0.8
  if(any(is.character(split), split < 0.5, split > 0.9)){
    stop("Input a split that is between 0.5 and 0.8")
  }
  # Ensure y and x matrices are valid
  if(is.null(data)){
    stop("No input data")
  }
  if(y_col %in% x_col){
    stop("response variable cannot also be a predictor")
  }
  if(length(y_col) != 1){
    stop("length of y_col must be 1")
  }
  if(is.numeric(y_col)){
    if(!(y_col %in% c(1:ncol(data)))){
      stop("y_col out of range")
    }
  }else if (is.character(y_col)){
    if(!(y_col %in% colnames(data))){
      stop("y_col not in dataframe")
    }
  }else{
    stop("y_col must be an integer or character")
  }
  if(!is.null(x_col)){
    if(all(is.numeric(x_col))){
      check_x <- 1:dim(data)[1]
    }else if(all(is.character(x_col))){
      check_x <- colnames(data)[colnames(data) != y_col]
    }else{
      stop("x_col must be a character vector or integer vector")
    }
    if(!(all(x_col %in% check_x))){
      stop("at least one predictor is not in dataframe")
    }
  }
  #Ensure model_type has been assigned
  if(call == "categorical_cv_split"){
    if(any(is.null(model_type), !(model_type %in% valid_models))){
      stop(sprintf("%s is an invalid model_type", model_type))
    }
    if(all(model_type == "logistic", length(levels(as.factor(data[,y_col]))) != 2)){
      stop("logistic regression requires a binary variable")
    }
  }
}
#Helper function for categorical_cv_split to check if additional arguments are valid
.check_additional_arguments <- function(model_type = NULL, ...){
  additional_args <- names(list(...))
  
  # Helper function to generate error message
  error_message <- function(model_name, invalid_args) {
    sprintf("The following arguments are invalid for %s or are incompatible with categorical_cv_split: %s",
            model_name, paste(invalid_args, collapse = ","))
  }
  
  # List of valid arguments for each model type
  valid_args_list <- list(
    "lda" = c("grouping", "prior", "method", "nu"),
    "qda" = c("grouping", "prior", "method", "nu"),
    "logistic" = c("weights", "start", "etastart", "mustart", "offset", "control", "contrasts", "intercept", "singular.ok", "type"),
    "svm" = c("scale", "type", "kernel", "degree", "gamma", "coef0", "cost", "nu", "class.weights", "cachesize", "tolerance", "epsilon", "shrinking", "cross", "probability", "fitted"),
    "naivebayes" = c("prior", "laplace", "usekernel", "usepoisson"),
    "ann" = c("weights", "size", "Wts", "mask", "linout", "entropy", "softmax", "censored", "skip", "rang", "decay", "maxit", "Hess", "trace", "MaxNWts", "abstol", "reltol"),
    "knn" = c("kmax", "ks", "distance", "kernel", "scale", "contrasts", "ykernel"),
    "decisiontree" = c("weights", "method", "parms", "control", "cost"),
    "randomforest" = c("ntree", "mtry", "weights", "replace", "classwt", "cutoff", "strata", "nodesize", "maxnodes", "importance", "localImp", "nPerm", "proximity", "oob.prox", "norm.votes", "do.trace", "keep.forest", "corr.bias", "keep.inbag")
  )
  
  valid_args <- valid_args_list[[model_type]]
  invalid_args <- additional_args[which(!additional_args %in% valid_args)]
  
  if (length(invalid_args) > 0) {
    stop(error_message(model_type, invalid_args))
  }
}

#Helper function for categorical_cv_split for stratified sampling
.stratified_sampling <- function(data,type, output, response_var, split = NULL, k = NULL,
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
           output[["sample_proportions"]][["split"]][["training"]] <- table(data[,response_var][output[["sample_indices"]][["split"]][["training"]]])/sum(table(data[,response_var][output[["sample_indices"]][["split"]][["training"]]]))
           #Store proportions of data  in test set
           output[["sample_proportions"]][["split"]][["test"]] <- table(data[,response_var][output[["sample_indices"]][["split"]][["test"]]])/sum(table(data[,response_var][output[["sample_indices"]][["split"]][["test"]]]))
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
           for(i in 1:k){
             #Keep initializing variable
             fold_idx <- c()
             output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
             #fold size; try to undershoot for excess
             fold_size <- floor(nrow(data)/k)
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
             output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
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
                   output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
                 }
               }
             }
           }
           #Output
           stratified_sampling_output <- list("output" = output)
         }
  )
}

#Helper function for .stratified_sampling to error check
.stratified_check <- function(class,class_indices,output,n){
  if(round(n*output[["class_proportions"]][[class]],0) == 0){
    stop(sprintf("0 indices selected for %s class\n not enough samples for stratified sampling",class))
  }
  if(round(n*output[["class_proportions"]][[class]],0) > length(class_indices[[class]])){
    stop(sprintf("not enough samples of %s class for stratified sampling",class))
  }
}
#Helper function for categorical_cv_split to remove unobserved data
.remove_obs <- function(trained_data,test_data,response_var,fold = NULL){
  check_predictor_levels <- list()
  for(col in colnames(trained_data[colnames(trained_data) != response_var])){
    if(is.character(trained_data[,col]) | is.factor(trained_data[,col])){
      check_predictor_levels[[col]] <- names(table(trained_data[,col]))[which(as.numeric(table(trained_data[,col])) != 0)]
    }
  }
  
  #Check new columns and set certain predictors in NA if the model has not been trained on
  for(col in colnames(test_data)[colnames(test_data) != response_var]){
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

