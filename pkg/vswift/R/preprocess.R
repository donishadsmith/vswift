# Helper function for classCV and genFolds to check if inputs are valid
.error_handling <- function(data, formula = NULL, target = NULL, predictors = NULL, models = NULL,
                            model_params = NULL, train_params = NULL, impute_params = NULL, save = NULL,
                            parallel_configs = NULL, create_data = NULL, call = NULL) {
  # List of valid inputs
  valid_inputs <- list(
    models = c(
      "lda", "qda", "logistic", "svm", "naivebayes", "nnet", "knn", "decisiontree",
      "randomforest", "multinom", "xgboost"
    ),
    imputes = c("knn_impute", "bag_impute")
  )

  # Create list of parameters
  if (call == "classCV") {
    params_list <- list(
      data = data, formula = formula, target = target, predictors = predictors, models = models,
      model_params = model_params, train_params = train_params, impute_params = impute_params,
      save = save
    )
  } else {
    params_list <- list(data = data, train_params = train_params, create_data = create_data)
    return(0)
  }

  # Check types
  for (param in names(params_list)) .param_checker(param, params_list[[param]])

  # Check formula and target
  if (inherits(c(formula, target), "NULL")) stop(sprintf("either `formula` or `target` must be specified"))

  if (!is.null(formula) & any(!is.null(target), !is.null(predictors))) {
    stop(sprintf("`formula` cannot be used when `target` or `predictors` are specified"))
  }

  # Check models
  error_msg <- "invalid model specified in `%s`, the following is a list of valid models: '%s'"

  if (!is.null(models) & !all(models %in% valid_inputs$models)) {
    stop(sprintf(error_msg, "models", paste(valid_inputs$models, collapse = "', '")))
  }

  # Check map_args
  map_args_models <- names(model_params$map_args)
  if (!is.null(map_args_models) & !all(map_args_models %in% valid_inputs$models)) {
    stop(sprintf(error_msg, "model_params$map_args", paste(valid_inputs$models, collapse = "', '")))
  }

  # Check vars
  .check_vars(formula, target, predictors, data)

  # Check map_args
  if (!is.null(model_params$map_args)) .check_args(model_params = model_params, call = "model")

  # Check logistic threshold
  obj <- c("reg:logistic", "binary:logistic", "binary:logitraw")

  if ("logistic" %in% models || "xgboost" %in% models && model_params$map_args$xgboost$params$objective %in% obj) {
    # Check if binary and threshold valid
    if (!is.null(formula)) target <- .get_var_names(formula = formula, data = data)$target
    binary_target <- length(levels(factor(data[, target], exclude = NA))) == 2
    valid_threshold <- model_params$logistic_threshold > 0 | model_params$logistic_threshold < 1

    if (!binary_target) {
      stop("'logistic' and 'xgboost' with a logistic regression objective requires a binary target")
    } else if (!valid_threshold) {
      stop("`threshold` must a numeric value from 0 to 1")
    }
  }

  # Check split, n_folds, final_model
  if (all(is.null(train_params$split), is.null(train_params$n_folds), is.null(model_params$final_model) || model_params$final_model == FALSE)) {
    stop("neither `split`, `n_folds`, or `final_model` specified")
  }


  if (!is.null(train_params$n_folds) && train_params$n_folds <= 2) stop("`train_params$n_folds` must greater than 2")

  if (!is.null(train_params$split) && c(train_params$split < 0 || train_params$split > 1)) {
    stop("`train_params$split` must a numeric value from 0 to 1")
  }

  # Check if impute method and args is valid
  if (!is.null(impute_params$method)) {
    if (!impute_params$method %in% valid_inputs$imputes) {
      stop(
        sprintf(
          "invalid method specified in `impute_params$method`, the following is a list of valid methods: '%s'",
          paste(valid_inputs$models, collapse = "', '")
        )
      )
    }

    if (!is.null(impute_params$args)) .check_args(impute_params = impute_params, call = "imputation")
  }

  # Check n_cores
  if (!is.null(parallel_configs$n_cores)) {
    if (is.null(train_params$n_folds)) stop("parallel processing is only available when `n_folds` is not NULL")
    if (parallel_configs$n_cores > as.vector(future::availableCores())) {
      stop(sprintf(
        "more cores specified than available; only %s cores available but %s cores specified",
        as.vector(future::availableCores()), parallel_configs$n_cores
      ))
    }
  }
}

# Helper function for to check if target and predictors are in dataframe
.check_vars <- function(formula, target, predictors, data) {
  if (!is.null(formula)) {
    vars <- .get_var_names(formula = formula, data = data)
  } else {
    vars <- list()
    vars$target <- target
    vars$predictors <- predictors
  }

  # Check target
  if (inherits(vars$target, c("numeric", "integer"))) {
    miss_target <- !vars$target %in% 1:ncol(data)
  } else {
    miss_target <- !vars$target %in% colnames(data)
  }

  if (miss_target) stop("specified target is not in the dataframe")

  # Check predictors
  if (!is.null(vars$predictors)) {
    if (inherits(vars$predictors, c("numeric", "integer"))) {
      pred_diff <- setdiff(vars$predictors, 1:ncol(data))
    } else {
      pred_diff <- setdiff(vars$predictors, colnames(data)[!colnames(data) == vars$target])
    }

    miss_pred <- ifelse(length(pred_diff) > 0, TRUE, FALSE)
    if (miss_pred) {
      stop(sprintf(
        "the following predictor indices or names were not found in the dataframe: '%s'",
        paste(pred_diff, collapse = "', '")
      ))
    }
  }
}

# Helper function for classCV to check if additional arguments are valid
.check_args <- function(model_params = NULL, impute_params = NULL, call) {
  # Helper function to generate error message
  error_message <- function(method_name, invalid_args) {
    sprintf(
      "The following arguments are invalid for %s or are incompatible with classCV: %s",
      method_name, paste(invalid_args, collapse = ", ")
    )
  }

  # List of valid arguments for each model type
  valid_args <- list(
    "model" = list(
      "lda" = c("prior", "method", "nu", "tol"),
      "qda" = c("prior", "method", "nu"),
      "logistic" = c("weights", "singular.ok", "maxit"),
      "svm" = c(
        "kernel", "degree", "gamma", "cost", "nu", "class.weights", "shrinking",
        "epsilon", "tolerance", "cachesize"
      ),
      "naivebayes" = c(
        "prior", "laplace", "usekernel", "bw", "kernal", "adjust", "weights",
        "give.Rkern", "subdensity", "from", "to", "cut"
      ),
      "nnet" = c(
        "size", "rang", "decay", "maxit", "softmax", "entropy", "abstol", "reltol", "Hess",
        "skip"
      ),
      "knn" = c("kmax", "ks", "distance", "kernel"),
      "decisiontree" = c("weights", "method", "parms", "control", "cost"),
      "randomforest" = c(
        "weights", "classwt", "ntree", "mtry", "nodesize", "importance", "localImp",
        "nPerm", "proximity", "keep.forest", "norm.votes"
      ),
      "multinom" = c("weights", "Hess"),
      "xgboost" = c(
        "params", "nrounds", "print_every_n", "feval", "verbose",
        "early_stopping_rounds", "obj", "save_period", "save_name"
      )
    ),
    "imputation" = list(
      "knn_impute" = c("neighbors"),
      "bag_impute" = c("trees", "seed_val")
    )
  )

  # Obtain user-specified models based on the number of models called
  methods <- ifelse(call == "model", names(model_params$map_args), impute_params$method)

  # Obtain user-specified model arguments based on the number of models called
  for (method in methods) {
    user_args <- ifelse(call == "model", names(model_params$map_args[[method]]), names(impute_params$args))
    invalid_args <- user_args[!user_args %in% valid_args[[call]][[method]]]

    # Special case
    if (method == "knn" & !"ks" %in% user_args) {
      warning("if `ks` not specified, knn may select a different optimal k for each fold")
    }

    if (length(invalid_args) > 0) stop(error_message(method, invalid_args))
  }
}

# Function to get name of target and features.
.get_var_names <- function(formula = NULL, target = NULL, predictors = NULL, data) {
  # Get target and predictor if formula used
  if (!is.null(formula)) {
    vars <- all.vars(formula)
    target <- vars[1]

    if ("." %in% vars[2:length(vars)]) {
      predictor_vec <- colnames(data)[colnames(data) != target]
    } else {
      predictor_vec <- vars[2:length(vars)]
    }
  } else {
    # Creating response variable
    target <- ifelse(is.character(target), target, colnames(data)[target])

    # Creating feature vector
    if (is.null(predictors)) {
      predictor_vec <- colnames(data)[colnames(data) != target]
    } else {
      predictor_vec <- ifelse(all(is.character(predictors)), predictors, colnames(data)[predictors])
    }
  }

  return(list("predictors" = predictor_vec, "target" = target))
}

.missing_summary <- function(data, target) {
  # Indices of unlabeled observations
  unlabeled_data <- which(is.na(data[, target]))
  # Number of observations with ONLY missing features
  incomplete_labeled_data <- unique(which(is.na(data), arr.ind = TRUE)[, "row"])
  n_incomplete_labeled_data <- length(incomplete_labeled_data[!incomplete_labeled_data %in% unlabeled_data])

  return(list("unlabeled_data_indices" = unlabeled_data, "n_incomplete_labeled_data" = n_incomplete_labeled_data))
}

.clean_data <- function(data, missing_info, imputation_requested) {
  perform_imputation <- imputation_requested

  # Messages
  msg1 <- sprintf("dropping %s unlabeled observations", length(missing_info$unlabeled_data_indices))

  if (imputation_requested) {
    msg2 <- sprintf(
      "%s labeled observations are missing data in one or more features and will be imputed",
      missing_info$n_incomplete_labeled_data
    )
  } else {
    msg2 <- sprintf(
      "dropping %s labeled observations with one or more missing features",
      length(missing_info$n_incomplete_labeled_data)
    )
  }

  # Dropping unlabeled data
  if (length(missing_info$unlabeled_data_indices) > 0) {
    warning(msg1)
    data <- data[-missing_info$unlabeled_data_indices, ]
  }

  # Issue warning for labeled data with missing values
  if (missing_info$n_incomplete_labeled_data > 0) warning(msg2)

  # Drop missing labeled data if no imputation requested
  if (missing_info$n_incomplete_labeled_data > 0 && !imputation_requested) {
    data <- data[complete.cases(data), ]
  }

  if (nrow(data) == sum(complete.cases(data)) && imputation_requested) {
    warning("remaining labeled observations has no missing data; imputation will not be performed")
    perform_imputation <- FALSE
  }
  return(list("cleaned_data" = data, "perform_imputation" = perform_imputation))
}

# Helper function to turn character data into factors
.convert_to_factor <- function(preprocessed_data, target, models, train_params) {
  # Make target factor, could be factor, numeric, or character
  preprocessed_data[, target] <- factor(preprocessed_data[, target])
  # Check for columns that are characters and factors
  columns <- colnames(preprocessed_data)[as.vector(sapply(
    preprocessed_data,
    function(x) is.character(x) | is.factor(x)
  ))]
  columns <- columns[!columns == target]
  # Create list to store levels for svm model
  col_levels <- list()

  # Turn character columns into factor
  for (col in columns) {
    preprocessed_data[, col] <- factor(preprocessed_data[, col])
    if ("svm" %in% models || train_params$remove_obs == TRUE) col_levels[[col]] <- levels(preprocessed_data[, col])
  }

  # Return output
  return(list("data" = preprocessed_data, "col_levels" = if (length(col_levels) > 0) col_levels else NULL))
}

# Function to retrieve columns that will be standardized
.get_cols <- function(df, standardize, target) {
  # Get predictor names
  predictors <- colnames(df)[colnames(df) != target]

  if (inherits(standardize, "logical")) {
    col_names <- predictors
  } else if (inherits(standardize, c("numeric", "integer"))) {
    # Remove any index value outside the range of the number of columns
    col_names <- predictors[intersect(1:ncol(df), standardize)]
    unused <- setdiff(standardize, 1:ncol(df))
    if (length(unused) > 0) {
      warning(sprintf("some indices are outside possible range and will be ignored: %s", paste(unused)))
    }
  } else {
    # Remove any column names not in dataframe
    unused <- intersect(standardize, predictors)
    col_names <- setdiff(standardize, predictors)
    if (length(unused) > 0) {
      warning(sprintf("some column names not in dataframe and will be ignored: %s", paste(unused)))
    }
  }
  return(col_names)
}

# Generate class specific information and initialize space for data partitions
.append_output <- function(target_vector, stratified = FALSE) {
  info_dict <- list()
  info_dict$class_summary <- list()
  info_dict$class_summary$classes <- names(table(factor(target_vector)))
  info_dict$data_partitions <- list("proportions" = NULL, "indices" = NULL, "dataframes" = NULL)

  if (stratified) info_dict$class_summary <- c(info_dict$class_summary, .get_class_info(target_vector))

  return(info_dict)
}

# Generate the iterations needed
.gen_iterations <- function(train_params, model_params) {
  iters <- c()
  if (!is.null(train_params$split)) iters <- "split"

  if (!is.null(train_params$n_folds)) iters <- c(iters, paste0("fold", 1:train_params$n_folds))

  if (model_params$final_model == TRUE) iters <- c(iters, "final")

  return(iters)
}

# Function to obtain numeric columns
.filter_cols <- function(df, col_names) {
  return(colnames(df)[sapply(df, is.numeric)])
}

# Function to standardize features for train data
.standardize_train <- function(train, test, standardize = TRUE, target) {
  col_names <- .get_cols(train, standardize, target)
  filtered_col_names <- .filter_cols(train, col_names)

  if (length(col_names) > 0) {
    for (col in filtered_col_names) {
      # Get mean and sample sd of the train data
      train_col_mean <- mean(as.numeric(train[, col]), na.rm = TRUE)
      train_col_sd <- sd(as.numeric(train[, col]), na.rm = TRUE)
      # Scale train and test data using the train mean and sample sd
      scaled_train_col <- (train[, col] - train_col_mean) / train_col_sd
      train[, col] <- as.vector(scaled_train_col)
      scaled_validation_col <- (test[, col] - train_col_mean) / train_col_sd
      test[, col] <- as.vector(scaled_validation_col)
    }
  } else {
    warning("no standardization has been done; either do to specified columns not being in dataframe or no columns
    being of class 'numeric'")
  }
  return(list("train" = train, "test" = test))
}

# Function to standardize features for preprocessed data/data used for final model
.standardize <- function(preprocessed_data, standardize = TRUE, target) {
  col_names <- .get_cols(preprocessed_data, standardize, target)
  filtered_col_names <- .filter_cols(preprocessed_data, col_names)

  if (length(col_names) > 0) {
    preprocessed_data[, filtered_col_names] <- as.data.frame(
      scale(preprocessed_data[, filtered_col_names], center = TRUE, scale = TRUE)
    )
  } else {
    warning("no standardization has been done; either do to specified columns not being in dataframe or no columns
    being of class 'numeric'")
  }

  return(list("preprocessed_data" = preprocessed_data))
}

# Function to standardize prior to creating or using the imputation model
.impute_standardize <- function(preprocessed_data = NULL, train = NULL, test = NULL, vars) {
  # Get data id
  use_data <- ifelse(!is.null(preprocessed_data), "preprocessed", "train")
  # Standardize
  if (use_data == "preprocessed") {
    data <- .standardize(preprocessed_data, target = vars$target)$preprocessed_data
    return("data" = data)
  } else {
    df_list <- .standardize_train(train, test, target = vars$target)
    data <- df_list$train
    test <- df_list$test
    rm(df_list)
    gc()
    return("data" = data, "test" = test, "use_data" = use_data)
  }
}

# Function to obtain the prep model
.impute_prep <- function(preprocessed_data = NULL, train = NULL, test = NULL, vars, impute_params) {
  # Standardize
  df_list <- .impute_standardize(preprocessed_data = NULL, train = NULL, test = NULL, vars)
  # Get data/output
  use_data <- df_list$use_data
  data <- df_list$data
  if (use_data == "train") test <- df_list$test

  # Create args list
  step_args <- list(recipe = recipes::recipe(~., data = data[, vars$predictors]))

  if (!is.null(impute_params$args)) {
    step_args <- c(step_args, impute_params$args, list(c(vars$predictors)))
  } else {
    step_args <- c(step_args, list(c(vars$predictors)))
  }

  # Prepare models & impute
  if (impute_params$method == "knn_impute") {
    step <- do.call(recipes::step_impute_knn, step_args)
  } else {
    step <- do.call(recipes::step_impute_bag, step_args)
  }

  prep <- recipes::prep(x = step, training = data[, vars$predictors], retain = FALSE)

  return(prep)
}

# Apply the prep model to data
.impute_bake <- function(preprocessed_data = NULL, train = NULL, test = NULL, vars, prep) {
  # Standardize
  df_list <- .impute_standardize(preprocessed_data = NULL, train = NULL, test = NULL, vars)
  # Get data/output
  use_data <- df_list$use_data
  data <- df_list$data
  if (use_data == "train") test <- df_list$test

  data <- cbind(data.frame(recipes::bake(prep, new_data = data[, vars$predictors])), subset(data, select = vars$target))

  if (use_data == "preprocessed") {
    return(list("preprocessed_data" = data))
  } else {
    test <- cbind(
      data.frame(recipes::bake(prep, new_data = test[, vars$predictors])),
      subset(test, select = vars$target)
    )
    return(list("train" = data, "test" = test))
  }
}
