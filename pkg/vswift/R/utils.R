# Function to partition data; indices is always the test set
.partition <- function(data, indices) {
  return(list("train" = data[-indices, ], "test" = data[indices, ]))
}

# Get test indices
.get_indices <- function(obj, id) {
  if (id == "split") {
    return(obj[[id]]$test)
  } else {
    # Should be foldn, such as fold1, fold2, etc
    return(obj$cv[[id]])
  }
}

# Unnest parallel list
.unnest <- function(par_list, iters, saved_mods = NULL) {
  targets <- c("metrics")
  metrics <- list()

  if (saved_mods == TRUE) {
    targets <- c("metrics", "models")
    models <- list()
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

  if (saved_mods == TRUE) {
    return(list("metrics" = metrics, "models" = models))
  } else {
    return(list("metrics" = metrics))
  }
}


# Function to prep the data for validation
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
    if (is_standardized == FALSE && kwargs$train_params$standardize == TRUE) {
      df_list <- .standardize_train(train, test, kwargs$train_params$standardize, target = kwargs$vars$target)
      train <- df_list$train
      test <- df_list$test
    }

    return("train" = train, "test" = test)
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
    if (is_standardized == FALSE && preproc_kwargs$standardize == TRUE) {
      df_list <- .standardize(preprocessed_data, standardize = TRUE, preproc_kwargs$vars$target)
    }

    return(df_list$preprocessed_data)
  }
}
