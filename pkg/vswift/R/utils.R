#Helper function for categorical_cv_split and stratified_split to check if inputs are valid
.error_handling <- function(data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = NULL, stratified = NULL,  random_seed = NULL,
                            call = NULL,...){
  #Valid models
  valid_models <- c("lda","qda","logistic","svm","naivebayes","ann","knn","decisiontree",
                    "randomforest")
  
  # Ensure data is not NULL
  if(is.null(data)){
    stop("no input data")
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
  
  if(call == "categorical_cv_split" || call == "stratified_split" & stratified == TRUE){
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
    }else{
      stop("target must be an integer or character")
    }
  }
  # Check if predictors are in data frame
  if(!is.null(predictors)){
    if(all(is.numeric(predictors))){
      check_x <- 1:dim(data)[1]
    }else if(all(is.character(predictors))){
      check_x <- colnames(data)[colnames(data) != target]
    }else{
      stop("predictors must be a character vector or integer vector")
    }
    if(!(all(predictors %in% check_x))){
      stop("at least one predictor is not in dataframe")
    }
  }
  
  #Ensure model_type has been assigned
  if(call == "categorical_cv_split"){
    if(any(is.null(model_type), !(model_type %in% valid_models))){
      stop(sprintf("%s is an invalid model_type", model_type))
    }
    if(model_type == "logistic" & any(length(levels(factor(data[,target], exclude = NA))) != 2, !is.numeric(threshold), threshold < 0.30 || threshold > 0.70)){
      if(length(levels(factor(data[,target], exclude = NA))) != 2){
        stop("logistic regression requires a binary variable")
      }else{
        stop("threshold must a numeric value from 0.30 to 0.70")
      }
    }
  }
}
# Helper function for categorical_cv_split to check if additional arguments are valid
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

#Helper function for categorical_cv_split to remove unobserved data
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
# Helper function to perform proper plotting depending on where command is ran
.check_env <- function(){
  system = as.character(Sys.info()["sysname"])
  if(Sys.getenv("RStudio") == "1"){
    new_window  <- ifelse(rstudioapi::isAvailable(), function(){placeholder = "placeholder"},ifelse(system == "Windows", windows, x11))
  }else{
    new_window <- ifelse(system == "Windows", windows, x11)
  }
  return(new_window)
}
# Helper function for regular plotting
.visible_plots <- function(object, split, cv){
  # Check if RStudio or GUI is running for proper plotting
  new_window <- vswift:::.check_env()
  if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
    # Plot metrics for training and test
    new_window()
    plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
    axis(1, at = 1:2, labels = c("Training","Test"))
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s F-Score", class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][["cv"]])[colnames(object[["metrics"]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      }else{
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Class: ",as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector), col = "red", lwd = 1)
      abline(h = mean(num_vector) + sd(num_vector), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector) - sd(num_vector), col = "blue", lty = 2, lwd = 1)
    }
  }
}
 
.dev_off_and_new <- function(){
  # Don't display plot if save_plot is TRUE
  if(all(Sys.getenv("RStudio") == "1",rstudioapi::isAvailable())){
    graphics.off()
  }else{
    dev.off()
  }
}

.save_plots <- function(object, path, split, cv, ...){
  if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
    # Save metrics for training and test
    png(filename = paste0(path,"train_test_classification_accuracy.png"),...)
    plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
    axis(1, at = 1:2, labels = c("Training","Test"))
    vswift:::.dev_off_and_new()
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_precision_%s.png",class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_recall_%s.png",class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_f-score_%s.png",  class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s F-Score", class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][["cv"]])[colnames(object[["metrics"]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        # Save metrics for cv
        png(filename = paste0(path,"cv_classification_accuracy.png"),...)
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      }else{
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        # Save metrics for cv
        png(filename = paste0(path,sprintf("cv_%s_%s.png",tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]])),...)
        file <- paste0(path,sprintf("cv_%s_%s.png",tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]]))
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Class: ",as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector), col = "red", lwd = 1)
      abline(h = mean(num_vector) + sd(num_vector), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector) - sd(num_vector), col = "blue", lty = 2, lwd = 1)
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
    }
  }
}


