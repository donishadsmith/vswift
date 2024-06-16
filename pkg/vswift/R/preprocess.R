#Helper function for classCV and genFolds to check if inputs are valid
#' @importFrom parallel detectCores
#' @noRd
#' @export
.error_handling <- function(formula = NULL, data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = NULL, stratified = NULL,  random_seed = NULL,
                            impute_method = NULL, impute_args = NULL, mod_args = NULL, n_cores = NULL, standardize = NULL, call = NULL, ...){
  
  # List of valid inputs
  valid_inputs <- list(valid_models = c("lda","qda","logistic","svm","naivebayes","ann","knn","decisiontree",
                                        "randomforest", "multinom", "gbm"),
                       valid_imputes = c("knn_impute","bag_impute"))
  
  
  # Check standardize
  if(!is.null(standardize)){
    if(!any(standardize == TRUE, standardize == FALSE, is.numeric(standardize), is.integer(standardize), is.character(standardize))){
      stop("`standardize` must either be TRUE, FALSE, or a numeric vector")
    }
  }
  # Check if impute method is valid
  if(!is.null(impute_method)){
    if(!impute_method %in% valid_inputs[["valid_imputes"]]){
      stop("invalid impute method")
    }
    # Check if impute_method is list
    if(!length(impute_method) == 1){
      stop("`impute_method` must be a character not list")
    }
    # Check if impute method is valid
    if(!is.null(impute_args)){
      if(all(impute_method == "knn_impute" | impute_method == "bag_impute", !inherits(impute_args, "list"))){
        stop("`impute_args` must be a list")
      }
      # Check if additional arguments are valid
      else if(impute_method == "knn_impute"| impute_method == "bag_impute"){
        .check_additional_arguments(impute_method = impute_method, impute_args = impute_args, call = "imputation")
      }
    }
    
  }
  
  if(all(length(model_type) > 1, length(list(...)) > 0)){
    stop("use `mod_args` parameter to specify model-specific arguments when calling multiple models")
  }
  
  if(!is.null(mod_args)){
    if(!inherits(mod_args, "list")){
      stop("`mod_args` must be a list")
    }
    if(length(model_type) == 1){
      stop("`mod_args`` used only when multiple models are specified")
    }
    if(!all(names(mod_args) %in% valid_inputs[["valid_models"]])){
      stop("invalid model in `mod_args`")
    }
  }
  
  if(!is.null(mod_args)){
    .check_additional_arguments(model_type = model_type, mod_args = mod_args, call = "multiple")
  }
  
  if(length(list(...)) > 0){
    .check_additional_arguments(model_type = model_type, call = "single", ...)
  }
  
  if (all(is.null(split), is.null(n_folds))){
    stop("Both `split` and `n_folds` cannot be null.")
  }
  
  # Ensure fold size is valid
  if(any(n_folds %in% c(0,1), n_folds < 0, n_folds > 30,is.character(n_folds), n_folds != as.integer(n_folds))){
    stop("`n_folds` must be a non-negative integer from 3-30")
  }
  
  # Ensure split is between 0.5 to 0.9
  if(any(is.character(split), split < 0.5, split > 0.9)){
    stop("`split` must be a numeric value from between 0.5 and 0.9")
  }
  
  # Get target and predictors if formula specified
  if(!is.null(formula)){
    if(any(!is.null(formula) & !is.null(target) || !is.null(predictors))){
      warning("`formula` specified with `target` and/or `predictors`, `formula` will overwrite the specified `target` and `predictors`")
    }
    get_features_target <- .get_features_target(formula = formula, data = data)
    target <- get_features_target[["target"]]
    predictors <- get_features_target[["predictor_vec"]]
  }
  
  # Ensure valid target variable
  if(call == "classCV" || call == "genFolds" & stratified == TRUE){
    # Ensure target is also not in predictors 
    if(target %in% predictors){
      stop("`target` cannot also be a `predictor`")
    }
    
    # Ensure there is only one target variable
    if(length(target) != 1){
      stop("length of `target` must be 1")
    }
    
    # Check if target is in dataframe
    if(is.numeric(target)){
      if(!(target %in% c(1:ncol(data)))){
        stop("`target` not in dataframe")
      }
    } else if (is.character(target)){
      if(!(target %in% colnames(data))){
        stop("`target` not in dataframe")
      }
    } else {
      stop("`target` must be an integer or character")
    }
  }
  
  # Check if predictors are in data frame
  if(!is.null(predictors)){
    if(all(is.numeric(predictors))){
      check_x <- 1:dim(data)[1]
    } else if (all(is.character(predictors))){
      check_x <- colnames(data)[colnames(data) != target]
    } else {
      stop("`predictors` must be a character vector or integer vector")
    }
    if(!(all(predictors %in% check_x))){
      stop("at least one predictor is not in dataframe")
    }
  }
  
  # Warning for knn
  
  if(all("knn" %in% model_type, !is.null(n_folds))){
    if(all(length(model_type) > 1, !is.null(mod_args))){
      if("knn" %in% names(mod_args)){
        check_ks <- ifelse("ks" %in% names(mod_args[["knn"]]), TRUE, FALSE)
      } else{
        check_ks <- FALSE
      }
    }
    else{
      check_ks <- ifelse("ks" %in% names(list(...)), TRUE, FALSE)
    }
    if(check_ks == FALSE){
      warning("if `ks` not specified, knn may select a different optimal k for each fold")
    }
  }
  
  #Ensure model_type has been assigned
  if(call == "classCV"){
    if(!inherits(model_type, "character")){
      stop("`model_type` must be a character or a vector containing characters")
    }
    if(!all(model_type %in% valid_inputs[["valid_models"]])){
      stop("invalid model in `model_type`")
    }
    if("logistic" %in% model_type & any(length(levels(factor(data[,target], exclude = NA))) != 2, !is.numeric(threshold), threshold < 0.30 || threshold > 0.70)){
      if(length(levels(factor(data[,target], exclude = NA))) != 2){
        stop("logistic regression requires a binary variable")
      } else {
        stop("`threshold` must a numeric value from 0.30 to 0.70")
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
#' @noRd
#' @export
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
#' @noRd
#' @export
.check_additional_arguments <- function(model_type = NULL, impute_method = NULL, impute_args = NULL, mod_args = NULL, call = NULL, ...){
  
  # Helper function to generate error message
  error_message <- function(method_name, invalid_args) {
    sprintf("The following arguments are invalid for %s or are incompatible with classCV: %s",
            method_name, paste(invalid_args, collapse = ","))
  }
  
  sub_list <- ifelse(call == "imputation", "imputation", "model")
  # List of valid arguments for each model type
  valid_args_list <- list(
    "model" = list("lda" = c("prior", "method", "nu"),
                   "qda" = c("prior", "method", "nu"),
                   "logistic" = c("weights","singular.ok", "maxit"),
                   "svm" = c("kernel", "degree", "gamma", "cost", "nu"),
                   "naivebayes" = c("prior", "laplace", "usekernel"),
                   "ann" = c("size", "rang", "decay", "maxit", "softmax", "entropy", "abstol", "reltol"),
                   "knn" = c("kmax", "ks", "distance", "kernel"),
                   "decisiontree" = c("weights", "method", "parms", "control", "cost"),
                   "randomforest" = c("weights", "ntree", "mtry", "nodesize", "importance"),
                   "multinom" = c("weights", "Hess"),
                   "gbm" = c("params", "nrounds")),
    "imputation" = list("knn_impute" = c("formula","neighbors"),
                        "bag_impute" = c("formula","trees"))
  )
  
  # Obtain user-specified models based on the number of models called
  if(call == "single" | call == "multiple"){
    methods <- model_type
  }  else {
    methods <- impute_method
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
    
    valid_args <- valid_args_list[[sub_list]][[method]]
    invalid_args <- additional_args[which(!additional_args %in% valid_args)]
    
    if(length(invalid_args) > 0) {
      stop(error_message(method, invalid_args))
    }
  }
}

# Function to get name of target and features.
#' @noRd
#' @export
.get_features_target <- function(formula = NULL, target = NULL, predictors = NULL, data){
  if(!is.null(formula)){
    vars <- all.vars(formula)
    target <- vars[1]
    predictor_vec <- vars[2:length(vars)]
    
    if("." %in% predictor_vec) predictor_vec <- colnames(data)[colnames(data) != target]
    
  } else{
    # Creating response variable
    target <- ifelse(is.character(target), target, colnames(data)[target])
    
    # Creating feature vector
    if(is.null(predictors)){
      predictor_vec <- colnames(data)[colnames(data) != target]
    } else {
      if(all(is.character(predictors))){
        predictor_vec <- predictors
      } else {
        predictor_vec <- colnames(data)[predictors]
      }
    }
  }
  
  get_features_target_output <- list("target" = target, "predictor_vec" = predictor_vec)
  
  return(get_features_target_output)
}

# Check if data is missing
#' @noRd
#' @export
.check_if_missing <- function(data){
  # Get rows of missing data
  miss <- sort(unique(which(is.na(data), arr.ind = TRUE)[,"row"]))
  # Warn users the total number of rows with at least one column of missing data
  if(length(miss) > 0){
    warning(sprintf("%s observations have at least one column of missing data", length(miss)))
    override_imputation <- FALSE
  }else{
    warning("no observations have missing data; no imputation will be performed")
    override_imputation <- TRUE
  }
  return(override_imputation)
}

# Helper function to remove missing data
#' @noRd
#' @export
.remove_missing_data <- function(data){
  
  # Warning for missing data if no imputation method selected or imputation fails to fill in some missing data
  miss <- nrow(data) - nrow(data[complete.cases(data),])
  if(miss  > 0){
    data <- data[complete.cases(data),]
    warning(sprintf("dataset contains %s observations with incomplete data only complete observations will be used"
                    ,paste(miss, collapse = ", ")))
  }
  
  remove_missing_data_output <- data
  # Return output
  return(remove_missing_data_output)
}


# Helper function to remove observations with missing target variable prior to imputation
#' @noRd
#' @export
.remove_missing_target <- function(data, target){
  missing_targets <- which(is.na(data[,target]))
  if(length(missing_targets) > 0){
    data <- data[-missing_targets,]
    warning(sprintf("the following observations have been removed due to missing target variable: %s"
                    ,paste(sort(missing_targets), collapse = ", ")))
  }
  return(data)
}

# Imputation function
#' @importFrom recipes step_impute_knn recipe all_predictors step_impute_bag prep bake
#' @noRd
#' @export
.imputation <- function(preprocessed_data, target, predictors, formula, imputation_method ,impute_args, classCV_output, iteration, parallel = TRUE, final = FALSE, random_seed = NULL){
  # Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  # Get column names
  col_names <- colnames(preprocessed_data)
  
  if(final == FALSE){
    # Get training and validation data
    if(iteration == "Training"){
      training_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["training"]],]
      validation_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["test"]],]
    } else {
      training_data <- preprocessed_data[-c(classCV_output[["sample_indices"]][["cv"]][[tolower(iteration)]]),]
      validation_data <- preprocessed_data[classCV_output[["sample_indices"]][["cv"]][[tolower(iteration)]],]
    }
    
    # Get names of rows
    training_rows <- rownames(training_data)
    validation_rows <- rownames(validation_data)

    if(imputation_method == "knn_impute"){
      if(!is.null(impute_args)){
        if(!is.null(impute_args[["formula"]])){
          formula <- impute_args[["formula"]]
        } else {
          formula <- formula
        }
        rec <- step_impute_knn(recipe = recipe(formula = formula, data = training_data), neighbors = impute_args[["neighbors"]], all_predictors())  
      } else {
        rec <- step_impute_knn(recipe = recipe(formula = formula, data = training_data),all_predictors())  
      }
    } else if(imputation_method == "bag_impute") {
      if(!is.null(impute_args)){
        if(!is.null(impute_args[["formula"]])){
          formula <- impute_args[["formula"]]
        } else {
          formula <- formula
        }
        rec <- step_impute_bag(recipe = recipe(formula = formula, data = training_data), trees = impute_args[["trees"]], all_predictors()) 
      } else {
        rec <- step_impute_bag(recipe = recipe(formula = formula, data = training_data), all_predictors())  
      }
    }
    
    prep <- prep(rec, training = training_data)
    # Apply the prepped recipe to the training data
    training_data_processed <- data.frame(bake(prep, new_data = training_data))
    
    # Create full data
    if(ncol(training_data_processed) != ncol(training_data)) training_data_processed <- cbind(training_data_processed, subset(training_data, select = col_names[!col_names %in% colnames(training_data_processed)]))[,col_names]
    
    
    # Apply the prepped recipe to the test data
    validation_data_processed <- data.frame(bake(prep, new_data = validation_data))
    
    # Create full data
    if(ncol(validation_data_processed) != ncol(validation_data)) validation_data_processed <- cbind(validation_data_processed, subset(validation_data, select = col_names[!col_names %in% colnames(validation_data_processed)]))[,col_names]
    
    # Update row names of the new processed data
    rownames(training_data_processed) <- training_rows
    rownames(validation_data_processed) <- validation_rows
    
    # Combine dataset and sort
    processed_data <- rbind(training_data_processed, validation_data_processed)
    sorted_rows <- as.character(sort(as.numeric(row.names(processed_data))))
    processed_data <- processed_data[sorted_rows,]
    
    # Create imputation_information list to store information 
    imputation_information <- .get_missing_info(training_data = training_data, validation_data = validation_data, iteration = iteration, imputation_method = imputation_method)
    
    if(iteration == "Training"){
      imputation_information[["split"]][["prep"]] <- prep
    } else {
      imputation_information[["cv"]][[tolower(iteration)]][["prep"]] <- prep
    }
    
    if(is.null(classCV_output[["imputation"]])){
      classCV_output[["imputation"]] <- imputation_information
    }
    
    if(parallel == FALSE){
      if(iteration != "Training"){
        if(is.null(classCV_output[["imputation"]][["cv"]])){
          classCV_output[["imputation"]][["cv"]] <- imputation_information[["cv"]]
        } else {
          classCV_output[["imputation"]][["cv"]][[tolower(iteration)]] <- imputation_information[["cv"]][[tolower(iteration)]]
        }
      }
    } 
    
    imputation_output <- list("processed_data" = processed_data, "classCV_output" = classCV_output)
    return(imputation_output)
    
  } else{
    # Get missing information
    imputation_information <- .get_missing_info(preprocessed_data = preprocessed_data, imputation_method = imputation_method)
    # Impute data
    if(imputation_method == "knn_impute"){
      if(!is.null(impute_args)){
        if(!is.null(impute_args[["formula"]])){
          formula <- impute_args[["formula"]]
        } else {
          formula <- formula
        }
        rec <- step_impute_knn(recipe = recipe(formula = formula, data = preprocessed_data), neighbors = impute_args[["neighbors"]], all_predictors())  
      } else {
        rec <- step_impute_knn(recipe = recipe(formula = formula, data = preprocessed_data), all_predictors())  
      }
    } else if(imputation_method == "bag_impute") {
      if(!is.null(impute_args)){
        if(!is.null(impute_args[["formula"]])){
          formula <- impute_args[["formula"]]
        } else {
          formula <- formula
        }
        rec <- step_impute_bag(recipe = recipe(formula = formula, data = preprocessed_data), trees = impute_args[["trees"]], all_predictors()) 
      } else {
        rec <- step_impute_bag(recipe = recipe(formula = formula, data = preprocessed_data), all_predictors())  
      }
    }
    
    prep <- prep(rec, data = preprocessed_data, new_data = NULL)
    processed_data <- data.frame(bake(prep, new_data = NULL))
    
    # Create full data
    if(ncol(processed_data) != ncol(preprocessed_data)) processed_data <- cbind(processed_data, subset(preprocessed_data, select = col_names[!col_names %in% colnames(processed_data)]))[,col_names]
    
    imputation_output <- list("processed_data" = processed_data, "classCV_output" = classCV_output)
    return(imputation_output)
  }
  
}


# Assist function for .imputation to get number of missing data for each column
#' @noRd
#' @export
.get_missing_info <- function(preprocessed_data = NULL, training_data = NULL, validation_data = NULL, iteration, imputation_method){
  # Create imputation list
  imputation_information <- list()
  
  imputation_information[["method"]] <- imputation_method
  
  # Create iteration vector
  if(is.null(preprocessed_data)){
    iter_vec <- preprocessed_data
  } else{
    iter_vec <- c(training_data, validation_data)
  }
  
  # Store information
  for(data in iter_vec){
    missing_cols <- colnames(data)[unique(as.vector(which(is.na(data),arr.ind = TRUE)[,"col"]))]
    missing_numbers <- lapply(missing_cols, function(x) length(which(is.na(data[,x]))))
    names(missing_numbers) <- missing_cols
    if(is.null(preprocessed_data)){
      imputation_information[[iteration]][[deparse(substitute(data))]][["missing_data"]] <- missing_numbers
    } else{
      imputation_information[["Final Model"]][["missing_data"]] <- missing_numbers
    }
  }
  
  return(imputation_information)
}

# Function to standardize data
#' @noRd
#' @export
.standardize <- function(training_data, validation_data, standardize, target){
  # Get predictor names
  predictors <- colnames(training_data)
  
  if(inherits(standardize, "logical")){
    col_names <- predictors
  } else if(inherits(standardize, c("numeric","integer"))){
    # Remove any index value outside the range of the number of columns
    n_cols <- 1:ncol(training_data)
    standardize <- standardize[which(standardize %in% n_cols)]
    unused <- standardize[which(!standardize %in% n_cols)]
    col_names <- predictors[standardize]
    if(length(unused) > 0){
      warning(sprintf("some indices are outside possible range and will be ignored: %s",paste(unused)))
    }
  } else{
    # Remove any column names not in dataframe
    unused <- standardize[which(!standardize %in% predictors)]
    col_names <- predictors[which(standardize %in% predictors)]
    if(length(unused) > 0){
      warning(sprintf("some column names not in dataframe and will be ignored: %s",paste(unused)))
    }
    
  }
  
  # Remove target
  col_names <- col_names[col_names != target]
  
  if(length(col_names) > 0){
    for(col in col_names){
      if(any(is.numeric(training_data[,col]), is.integer(training_data[,col]))){
        # Get mean and sample sd of the training data
        training_col_mean <- mean(as.numeric(training_data[,col]))
        training_col_sd <- sd(as.numeric(training_data[,col]))
        # Scale training and test data using the training mean and sample sd
        scaled_training_col <- (training_data[,col] - training_col_mean)/training_col_sd
        training_data[,col] <- as.vector(scaled_training_col)
        scaled_validation_col <- (validation_data[,col] - training_col_mean)/training_col_sd
        validation_data[,col] <- as.vector(scaled_validation_col)
      }
    }
  } else{
    warning("no standardization has been done; standardization specified but column indices are outside possible range or column names don't exist")
  }
  
  standardize_list <- list("training_data" = training_data, "validation_data" = validation_data)
  return(standardize_list)
}

