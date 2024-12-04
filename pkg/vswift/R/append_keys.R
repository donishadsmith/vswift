# Function for appending missing keys to parameters that are lists
.append_keys <- function(param, struct, models = NULL, call = "classCV", ...) {
  if (call != "classCV") {
    train_keys <- list("split" = NULL, "n_folds" = NULL, "stratified" = FALSE, "random_seed" = NULL)
  } else {
    train_keys <- list(
      "split" = NULL, "n_folds" = NULL, "stratified" = FALSE, "random_seed" = NULL,
      "standardize" = FALSE, "remove_obs" = FALSE
    )
  }

  default_keys <- list(
    "model_params" = list("map_args" = NULL, "logistic_threshold" = 0.5, "final_model" = FALSE),
    "train_params" = train_keys,
    "impute_params" = list("method" = NULL, "args" = NULL),
    "save" = list("models" = FALSE, "data" = FALSE),
    "parallel_configs" = list("n_cores" = NULL, "future.seed" = NULL)
  )

  # Ensure struct is a list
  if (!inherits(struct, "list")) stop(sprintf("`%s` must be a list", param))

  # Ensure struct is a nested list
  if (is.null(names(struct))) {
    stop(
      sprintf(
        "`%s` must be a nested list containing a valid key: '%s'", param,
        paste(names(default_keys[[param]]), collapse = "', '")
      )
    )
  }

  # Drop keys
  drop_keys <- setdiff(names(struct), names(default_keys[[param]]))

  if (length(drop_keys) > 0) {
    warning(
      sprintf(
        "the following keys are invalid for `%s` and will be ignored: '%s'", param,
        paste(drop_keys, collapse = "', '")
      )
    )
    struct <- struct[!names(struct) %in% drop_keys]
  }

  # Get missing keys
  miss_keys <- setdiff(names(default_keys[[param]]), names(struct))

  # Append missing keys
  if (length(miss_keys) > 0) {
    new_struct <- c(struct, default_keys[[param]][names(default_keys[[param]]) %in% miss_keys])
  } else {
    new_struct <- struct
  }

  if (param == "model_params") {
    # Append ellipses
    if (length(models) == 1 && length(list(...)) > 0) {
      new_struct$map_args[[models]] <- list(...)
    }

    gbm_logistic <- c("reg:logistic", "binary:logistic", "binary:logitraw")

    if (!"logistic" %in% models && !("gbm" %in% models && new_struct$map_args$gbm$params$objective %in% gbm_logistic)) {
      # Assigning NULL to logistic_threshold directly will delete this key and leave an empty space
      new_struct <- c(new_struct[!names(new_struct) == "logistic_threshold"], list(logistic_threshold = NULL))
    }
  }

  # Order
  return(new_struct)
}
