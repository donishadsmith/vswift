# Function for appending missing keys to parameters that are lists
.append_param_keys <- function(param, struct, models = NULL, call = "classCV", ...) {
  # Evaluate to get default keys for specific parameters
  default_keys <- eval(as.list(.DEFAULT_KEYS[[2]])[[param]])

  # Ensure struct is a list
  if (!inherits(struct, "list")) stop(sprintf("`%s` must be a list", param))

  # Ensure struct is a nested list
  if (is.null(names(struct))) {
    stop(
      sprintf(
        "`%s` must be a nested list containing one of the following valid keys: '%s'", param,
        paste(names(default_keys), collapse = "', '")
      )
    )
  }

  # Drop keys
  drop_keys <- setdiff(names(struct), names(default_keys))

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
  miss_keys <- setdiff(names(default_keys), names(struct))

  # Append missing keys
  if (length(miss_keys) > 0) {
    new_struct <- c(struct, default_keys[names(default_keys) %in% miss_keys])
  } else {
    new_struct <- struct
  }

  if (param == "model_params") {
    # Append ellipses
    if (length(models) == 1 && length(list(...)) > 0) new_struct$map_args[[models]] <- list(...)

    has_lambda <- (
      length(new_struct$map_args$regularized_logistic$lambda == 1) ||
        length(new_struct$map_args$regularized_multinomial$lambda == 1)
    )

    if (!any(c("regularized_logistic", "regularized_multinomial") %in% models) || has_lambda) {
      new_struct <- c(new_struct[!names(new_struct) %in% c("rule", "verbose")], list(rule = NULL, verbose = NULL))
    }
  }

  # Order
  return(new_struct[names(default_keys)])
}
