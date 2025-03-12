# Helper function to partition data; indices is always the test set
.partition <- function(data, indices) {
  return(list("train" = data[-indices, ], "test" = data[indices, ]))
}

# Helper function to generate all dataframes; primarily used if save$data = TRUE
.create_data <- function(data, subsets) {
  df_list <- list()
  # Get data for train-test split
  if ("split" %in% names(subsets)) {
    df_list$split$train <- data[subsets$split$train, ]
    df_list$split$test <- data[subsets$split$test, ]
  }

  # Get train and test partitions for cv
  if ("cv" %in% names(subsets)) {
    for (fold in names(subsets$cv)) {
      df_list$cv[[fold]]$train <- data[-subsets$cv[[fold]], ]
      df_list$cv[[fold]]$test <- data[subsets$cv[[fold]], ]
    }
  }

  return(df_list)
}

# Helper function to get test indices
.get_indices <- function(obj, id) {
  if (id == "split") {
    return(obj[[id]]$test)
  } else {
    # Should be foldn, such as fold1, fold2, etc
    return(obj$cv[[id]])
  }
}

# Helper function to generate foldid
.get_foldid <- function(cv_indxs, N) {
  # Replace fold ids with numbers
  names(cv_indxs) <- 1:length(cv_indxs)
  # Get the fold number of each observation
  foldid <- as.vector(sapply(1:N, function(indx) .get_key(indx, cv_indxs)))

  # Returns vector where each position has an id, corresponding to the fold an observation belongs to
  return(foldid)
}

# Helper function to obtain key of a value in a sublist
.get_key <- function(indx, cv_indxs) {
  # Get a bool vector that indicates which sublist has the indx/observation
  bool_vec <- sapply(cv_indxs, function(x) indx %in% x)

  # Return fold id for index
  return(as.numeric(names(bool_vec)[bool_vec]))
}

# Helper function to unnest parallel list
.unnest <- function(par_list, iters, model, saved_mods) {
  targets <- c("metrics")
  metrics <- list()
  lambdas <- c()

  if (isTRUE(saved_mods)) {
    targets <- c("metrics", "models")
    models <- list()
  }

  # Append the optimal lambdas; use c() to retain names
  if (startsWith(model, "regularized")) {
    for (i in seq_along(iters)) lambdas <- c(lambdas, par_list[[i]]$optimal_lambda)
  }

  for (target in targets) {
    for (i in seq_along(iters)) {
      if (target == "metrics") {
        if (iters[i] == "split") {
          metrics$split <- par_list[[i]]$metrics$split
        } else {
          metrics$cv <- c(metrics$cv, par_list[[i]]$metrics$cv)
        }
      } else {
        if (iters[i] == "split") {
          models$split <- par_list[[i]]$models$split
        } else {
          models$cv <- c(models$cv, par_list[[i]]$models$cv)
        }
      }
    }
  }

  out <- list("metrics" = metrics)

  if (isTRUE(saved_mods)) out$models <- models

  if (length(lambdas) > 0) out$optimal_lambdas <- lambdas

  return(out)
}


# Helper function to prep the data for validation
.prep_data <- function(i = NULL, train = NULL, test = NULL, kwargs = NULL, preprocessed_data = NULL, preproc_kwargs = NULL) {
  is_standardized <- FALSE

  if (is.null(preprocessed_data)) {
    # Impute; determine if impute_models is not NULL or an empty list
    if (!is.null(kwargs$impute_models) && length(kwargs$impute_models) > 0) {
      df_list <- .impute_bake(train = train, test = test, vars = kwargs$vars, prep = kwargs$impute_models[[i]])
      train <- df_list$train
      test <- df_list$test
      is_standardized <- TRUE
    }

    # Standardize
    if (isFALSE(is_standardized) && isTRUE(kwargs$train_params$standardize)) {
      df_list <- .standardize_train(train, test, kwargs$train_params$standardize, target = kwargs$vars$target)
      train <- df_list$train
      test <- df_list$test
    }

    return(list("train" = train, "test" = test))
  } else {
    # Impute; determine if impute_models is not NULL or an empty list
    if (!is.null(preproc_kwargs$prep)) {
      df_list <- .impute_bake(
        preprocessed_data = preprocessed_data,
        vars = preproc_kwargs$vars, prep = preproc_kwargs$prep
      )
      preprocessed_data <- df_list$preprocessed_data
      is_standardized <- TRUE
    }

    # Standardize
    if (isFALSE(is_standardized) && isTRUE(preproc_kwargs$standardize)) {
      df_list <- .standardize(preprocessed_data, standardize = TRUE, preproc_kwargs$vars$target)
    }

    return(df_list$preprocessed_data)
  }
}


# Helper function for to remove observations in test set with factors in predictors not observed during train
.remove_obs <- function(train, test, col_levels, id) {
  # Iterate over columns and check for the factors that exist in test set but not the train set
  for (col in names(col_levels)) {
    delete_rows <- which(!test[, col] %in% train[, col])
    obs <- row.names(test)[delete_rows]

    if (length(obs) > 0) {
      warning(sprintf(
        "for predictor `%s` in `%s` data partition has at least one class the model has not trained on\nthese observations will be temporarily removed: %s",
        col, id, paste(obs, collapse = ",")
      ))
      test <- test[-delete_rows, ]
    }
  }

  return(list("test" = test))
}

# Helper function to get models present in vswift object
.intersect_models <- function(x, models) {
  # Get models
  if (is.null(models)) {
    models <- x$configs$models
  } else {
    # Make lowercase
    models <- sapply(models, function(x) tolower(x))
    models <- intersect(models, x$configs$models)

    if (length(models) == 0) stop("no valid models specified in `models`")

    # Warning when invalid models specified
    invalid_models <- models[which(!models %in% models)]
    if (length(invalid_models) > 0) {
      warning(sprintf(
        "invalid model in models or information for specified model not present in vswift x: %s",
        paste(unlist(invalid_models), collapse = ", ")
      ))
    }
  }

  return(models)
}


# Helper function to convert matrices to vectors
# Handle prediction output, some models will produce a matrix with posterior probabilities for binary outcomes
.tovec <- function(model, result, keys) {
  convert <- (
    !(model %in% c("logistic", "regularized_logistic", "nnet", "multinom", "xgboost")) &&
      !is.null(keys) && length(dim(result)) == 2
  )

  if (convert) result <- result[, names(keys)[keys == 1]]

  return(as.vector(result))
}

# Helper function to determine if default boundary should be used
.determine_threshold <- function(model, obj, threshold, issue_warning = TRUE) {
  xgboost_logistic <- c("reg:logistic", "binary:logistic", "binary:logitraw")
  check_bool <- obj %in% xgboost_logistic

  if ((model == "logistic" || (model == "xgboost" && isTRUE(check_bool))) && is.null(threshold)) {
    threshold <- 0.5
    if (issue_warning) warning(sprintf("using a default threshold of 0.5 to classify groups for %s model", model))
  } else if (model == "xgboost" && obj == "binary:hinge") {
    threshold <- NULL
  }

  return(threshold)
}

# Helper function to ensure all columns have the same levels as the original data for svm
.relevel_cols <- function(data, col_levels) {
  data[, names(col_levels)] <- data.frame(
    lapply(names(col_levels), function(col) factor(data[, col], levels = col_levels[[col]]))
  )

  return(data)
}
