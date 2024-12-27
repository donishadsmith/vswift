# Function to perform sequential validation
.sequential <- function(kwargs) {
  # Create lists
  met_list <- list()
  mod_list <- list()

  for (i in kwargs$iters) {
    # Get indices
    test_indices <- .get_indices(kwargs$indices, i)
    # Get training and validation data
    df_list <- .partition(kwargs$preprocessed_data, test_indices)
    # Get train and test data
    train <- df_list$train
    test <- df_list$test
    # Prep data
    if (kwargs$train_params$standardize == TRUE || !is.null(kwargs$impute_models) && length(kwargs$impute_models) > 0) {
      df_list <- .prep_data(i, train, test, kwargs)
      train <- df_list$train
      test <- df_list$test
    }

    out <- .validation_entry(i, df_list$train, df_list$test, kwargs)

    # Append metrics and models list
    if (i == "split") {
      met_list$split <- out$metrics$split
      if (kwargs$save_mods) mod_list$split <- out$models$split
    } else {
      met_list$cv <- c(met_list$cv, out$metrics$cv)
      if (kwargs$save_mods) mod_list$cv <- c(mod_list$cv, out$models$cv)
    }
  }

  if (length(mod_list) > 0) {
    return(list("metrics" = met_list, "models" = mod_list))
  } else {
    return(list("metrics" = met_list))
  }
}

# Function to perform validation of split and folds using parallel processing while combining outputs
#' @importFrom future.apply future_lapply
.parallel <- function(kwargs, parallel_configs, iters) {
  future::plan(future::multisession, workers = parallel_configs$n_cores)
  par_out <- future_lapply(iters, function(i) {
    # Get indices
    test_indices <- .get_indices(kwargs$indices, i)
    # Get training and validation data
    df_list <- .partition(kwargs$preprocessed_data, test_indices)
    # Remove data; Parallel processing will create copies of data to avoid race conditions if data not shared
    kwargs$preprocessed_data <- NULL

    # Get train and test data
    train <- df_list$train
    test <- df_list$test

    # Prep data
    if (kwargs$train_params$standardize == TRUE || !is.null(kwargs$impute_models) && length(kwargs$impute_models) > 0) {
      df_list <- .prep_data(i, train, test, kwargs)
      train <- df_list$train
      test <- df_list$test
    }

    # Get output
    out <- .validation_entry(i, train, test, kwargs)
  }, future.seed = if (!is.null(parallel_configs$future.seed)) parallel_configs$future.seed else TRUE)

  # Close the background workers
  future::plan(future::sequential)

  par_unnest <- .unnest(par_out, iters, kwargs$save_mods)
  return(par_unnest)
}

# Entry point for validation
.validation_entry <- function(i, train, test, kwargs) {
  # Create lists
  met_list <- list()
  mod_list <- list()

  name <- ifelse(i == "split", i, "cv")

  val_out <- .validation(
    i, train, test, kwargs$model, kwargs$formula, kwargs$model_params, kwargs$vars,
    kwargs$train_params$remove_obs, kwargs$col_levels, kwargs$class_summary$classes, kwargs$class_summary$keys,
    kwargs$met_df[[kwargs$model]][[name]], kwargs$train_params$random_seed, kwargs$save_mods
  )

  if (name == "split") {
    met_list$split <- val_out$met_df
    if (kwargs$save_mods == TRUE) mod_list$split <- val_out$train_mod
  } else {
    met_list$cv[[i]] <- val_out$met_df
    if (kwargs$save_mods == TRUE) mod_list$cv[[i]] <- val_out$train_mod
  }

  if (kwargs$save_mods == FALSE) {
    return(list("metrics" = met_list))
  } else {
    return(list("metrics" = met_list, "models" = mod_list))
  }
}
