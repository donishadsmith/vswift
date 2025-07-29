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
    miss_target <- !vars$target %in% seq_len(ncol(data))
  } else {
    miss_target <- !vars$target %in% colnames(data)
  }

  if (miss_target) stop("specified target is not in the dataframe")

  # Check predictors
  if (!is.null(vars$predictors)) {
    if (inherits(vars$predictors, c("numeric", "integer"))) {
      pred_diff <- setdiff(vars$predictors, seq_len(ncol(data)))
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
.check_args <- function(model_params = NULL, impute_params = NULL, caller) {
  # Helper function to generate error message
  error_message <- function(method_name, invalid_args) {
    sprintf(
      "The following arguments are invalid for %s or are incompatible with classCV: %s",
      method_name, paste(invalid_args, collapse = ", ")
    )
  }

  # Obtain user-specified models based on the number of models called
  methods <- ifelse(caller == "model", names(model_params$map_args), impute_params$method)

  # Obtain user-specified model arguments based on the number of models called
  for (method in methods) {
    user_args <- ifelse(caller == "model", names(model_params$map_args[[method]]), names(impute_params$args))
    invalid_args <- user_args[!user_args %in% .VALID_ARGS[[caller]][[method]]]

    # Special case
    if (method == "knn" && !"ks" %in% user_args) {
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
  # Get only features to identify observations where all features are missing
  feature_cols <- colnames(data)[colnames(data) != target]
  feature_data <- data[, feature_cols]
  # Indices with all features missing
  missing_all_features_indices <- which(
    sapply(seq(nrow(feature_data)), function(x) all(is.na(feature_data[x, feature_cols])))
  )
  # Number of observations with ONLY missing features
  incomplete_labeled_data <- unique(which(is.na(data), arr.ind = TRUE)[, "row"])
  incomplete_labeled_data <- incomplete_labeled_data[!incomplete_labeled_data %in% missing_all_features_indices]
  n_incomplete_labeled_data <- length(incomplete_labeled_data[!incomplete_labeled_data %in% unlabeled_data])

  return(
    list(
      "unlabeled_data_indices" = unlabeled_data,
      "missing_all_features_indices" = missing_all_features_indices,
      "n_incomplete_labeled_data" = n_incomplete_labeled_data
    )
  )
}

.clean_data <- function(data, missing_info, imputation_requested, issue_warning = TRUE) {
  perform_imputation <- imputation_requested

  msg1 <- sprintf("dropping %s unlabeled observations", length(missing_info$unlabeled_data_indices))
  msg2 <- sprintf("dropping %s observations with all features missing", length(missing_info$missing_all_features_indices))

  if (imputation_requested) {
    msg3 <- sprintf(
      "%s labeled observations are missing data in one or more features and will be imputed",
      missing_info$n_incomplete_labeled_data
    )
  } else {
    msg3 <- sprintf(
      "dropping %s labeled observations with one or more missing features",
      length(missing_info$n_incomplete_labeled_data)
    )
  }

  # Dropping unlabeled data
  if (length(missing_info$unlabeled_data_indices) > 0 || length(missing_info$missing_all_features_indices) > 0) {
    if (isTRUE(issue_warning) & length(missing_info$unlabeled_data_indices) > 0) warning(msg1)
    if (isTRUE(issue_warning) & length(missing_info$missing_all_features_indices) > 0) warning(msg2)
    discard_indices <- c(missing_info$unlabeled_data_indices, missing_info$missing_all_features_indices)
    data <- data[-discard_indices, ]
  }

  # Issue warning for labeled data with missing values
  if (missing_info$n_incomplete_labeled_data > 0) warning(msg3)

  # Drop missing labeled data if no imputation requested
  if (missing_info$n_incomplete_labeled_data > 0 && !imputation_requested) {
    data <- data[complete.cases(data), ]
  }

  if (nrow(data) == sum(complete.cases(data)) && imputation_requested) {
    if (isTRUE(issue_warning)) warning("remaining labeled observations has no missing data; imputation will not be performed")
    perform_imputation <- FALSE
  }

  return(list("cleaned_data" = data, "perform_imputation" = perform_imputation))
}

# Helper function to turn character data into factors
.convert_to_factor <- function(preprocessed_data, target, models, remove_obs = FALSE) {
  # Make target factor, could be factor, numeric, or character
  preprocessed_data[, target] <- factor(preprocessed_data[, target])
  # Check for columns that are characters and factors
  cols <- colnames(preprocessed_data)[as.vector(sapply(
    preprocessed_data,
    function(x) is.character(x) | is.factor(x)
  ))]

  # Only retain features
  cols <- cols[!cols == target]

  # Convert to data.table
  preprocessed_dt <- data.table(preprocessed_data)
  # Turn character columns into factor
  preprocessed_dt[, (cols) := Map(function(x) factor(x), .SD), .SDcols = cols]

  # Create list to store levels for svm model or to remove observations later
  if ("svm" %in% models || isTRUE(remove_obs)) {
    # Sapply through each column and collect levels
    col_levels <- sapply(preprocessed_dt[, .SD, .SDcols = cols], levels)
  }

  # Revert back to data.frame
  preprocessed_data <- as.data.frame(preprocessed_dt)

  return(list(
    "data" = preprocessed_data,
    "col_levels" = if (exists("col_levels") && length(col_levels) > 0) col_levels else NULL
  ))
}

# Function to retrieve columns that will be standardized
.get_cols <- function(data, standardize, target) {
  # Get predictor names
  predictors <- colnames(data)[colnames(data) != target]

  if (inherits(standardize, "logical")) {
    col_names <- predictors
  } else if (inherits(standardize, c("numeric", "integer"))) {
    # Remove any index value outside the range of the number of columns
    col_names <- predictors[intersect(1:ncol(data), standardize)]
    unused <- setdiff(standardize, 1:ncol(data))
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

  if (isTRUE(model_params$final_model)) iters <- c(iters, "final")

  return(iters)
}

# Function to obtain numeric columns
.filter_cols <- function(data, col_names) {
  return(colnames(data)[sapply(data, is.numeric)])
}

# Function to restore rownames of original data
.restore_rownames <- function(data, rownames) {
  rownames(data) <- rownames

  return(data)
}

# Function to standardize features for train data
.standardize_train <- function(train, test = NULL, standardize = TRUE, target, caller = "standard") {
  col_names <- .get_cols(train, standardize, target)
  filtered_col_names <- .filter_cols(train, col_names)

  train_dt <- data.table(train)

  if (length(col_names) > 0) {
    # Obtain training mean and sd of all numeric columns
    means <- train_dt[, lapply(.SD, function(x) mean(x, na.rm = TRUE)), .SDcols = filtered_col_names]
    stdevs <- train_dt[, lapply(.SD, function(x) sd(x, na.rm = TRUE)), .SDcols = filtered_col_names]
    # Standardize in place
    train_dt[, (filtered_col_names) := Map(function(x, m, s) (x - m) / s, .SD, means, stdevs), .SDcols = filtered_col_names]
    # Standardize test
    if (caller != ".impute_prep") {
      test_dt <- data.table(test)
      # Standardize using training parameters
      test_dt[, (filtered_col_names) := Map(function(x, m, s) (x - m) / s, .SD, means, stdevs), .SDcols = filtered_col_names]
    }
  } else {
    warning("no standardization has been done; either do to specified features not being in dataframe or no features
    being of class 'numeric'")
  }

  if (caller != ".impute_prep") {
    return(list(
      "train" = .restore_rownames(as.data.frame(train_dt), row.names(train)),
      "test" = .restore_rownames(as.data.frame(test_dt), row.names(test))
    ))
  } else {
    return(list("train" = .restore_rownames(as.data.frame(train_dt), row.names(train))))
  }
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
    warning("no standardization has been done; either do to features columns not being in dataframe or no features
    being of class 'numeric'")
  }

  return(list("preprocessed_data" = preprocessed_data))
}

# Function to standardize prior to creating or using the imputation model
.impute_standardize <- function(preprocessed_data = NULL, train = NULL, test = NULL, vars, caller = "standard") {
  # Get data id
  use_data <- ifelse(!is.null(preprocessed_data), "preprocessed", "train")

  if (use_data == "preprocessed") {
    data <- .standardize(preprocessed_data = preprocessed_data, target = vars$target)$preprocessed_data
    return(list("data" = data, "use_data" = use_data))
  } else {
    df_list <- .standardize_train(train = train, test = test, target = vars$target, caller = caller)

    if (caller != ".impute_prep") {
      return(list("data" = df_list$train, "test" = df_list$test, "use_data" = use_data))
    } else {
      return(list("data" = df_list$train))
    }
  }
}

# Function to obtain the prep model
.impute_prep <- function(preprocessed_data = NULL, train = NULL, vars, impute_params) {
  # Standardize
  data <- .impute_standardize(
    preprocessed_data = preprocessed_data, train = train, vars = vars,
    caller = ".impute_prep"
  )$data

  # Create args list
  step_args <- list(recipe = recipes::recipe(~., data = data[, vars$predictors]))

  if (!is.null(impute_params$args)) {
    step_args <- c(step_args, impute_params$args, list(c(vars$predictors)))
  } else {
    step_args <- c(step_args, list(c(vars$predictors)))
  }

  # Prepare models & impute
  if (impute_params$method == "impute_bag") {
    step <- do.call(recipes::step_impute_bag, step_args)
  } else {
    step <- do.call(recipes::step_impute_knn, step_args)
  }

  prep <- recipes::prep(x = step, training = data[, vars$predictors], retain = FALSE)

  return(prep)
}

# Apply the prep model to data
.impute_bake <- function(preprocessed_data = NULL, train = NULL, test = NULL, vars, prep) {
  if (!is.null(preprocessed_data)) {
    df_list <- .impute_standardize(preprocessed_data = preprocessed_data, vars = vars)
  } else {
    df_list <- .impute_standardize(train = train, test = test, vars = vars)
  }

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
