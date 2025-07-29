# Helper function for classCV and genFolds to check if inputs are valid
.error_handling <- function(data, formula = NULL, target = NULL, predictors = NULL, models = NULL,
                            model_params = NULL, train_params = NULL, impute_params = NULL, save = NULL,
                            parallel_configs = NULL, create_data = NULL, caller = NULL) {
  valid_models <- names(.VALID_ARGS$model)
  valid_imputes <- names(.VALID_ARGS$imputation)

  # Create list of parameters
  if (caller == "classCV") {
    params_list <- list(
      data = data, formula = formula, target = target, predictors = predictors, models = models,
      train_params = train_params, model_params = model_params, impute_params = impute_params,
      save = save, parallel_configs = parallel_configs
    )
  } else {
    params_list <- list(data = data, target = target, train_params = train_params, create_data = create_data)
  }

  # Check types
  for (param in names(params_list)) .type_validator(param, params_list[[param]])

  # Determine to stop execution
  .stop_execution(train_params, model_params, caller)

  if (!is.null(train_params$n_folds) && train_params$n_folds <= 2) stop("`train_params$n_folds` must greater than 2")

  if (!is.null(train_params$split) && c(train_params$split < 0 || train_params$split > 1)) {
    stop("`train_params$split` must a numeric value from 0 to 1")
  }

  # Check formula and target
  msg <- ifelse(caller == "classCV", "either `formula` or `target` must be specified", "`target` must be specified")
  if (inherits(c(formula, target), "NULL")) stop(msg)

  # Check vars
  .check_vars(formula, target, predictors, data)

  # Exit early for genFolds
  if (caller == "genFolds") {
    return(0)
  }

  # Check that only formula and target are specified
  if (!is.null(formula) && any(!is.null(target), !is.null(predictors))) {
    stop(sprintf("`formula` cannot be used when `target` or `predictors` are specified"))
  }

  # Check models
  error_msg <- "invalid model specified in `%s`, the following is a list of valid models: '%s'"

  if (!is.null(models) && !all(models %in% valid_models)) {
    stop(sprintf(error_msg, "models", paste(valid_models, collapse = "', '")))
  }

  # Check map_args
  .check_map_args(model_params, valid_models, error_msg)

  # Check rule
  if (any(c("regularized_logistic", "regularized_multinomial") %in% models) && !is.null(model_params$rule)) {
    intersect_char <- intersect(c("min", "1se"), model_params$rule)
    if (length(intersect_char) == 0) stop("'min' and '1se' are the only valid options for `model_params$rule`")
  }

  # Check if target is binary
  .check_binary_models(data, formula, target, models, model_params)

  # Check if impute method and args is valid
  .check_imputes(valid_imputes, impute_params)

  # Check n_cores
  .check_cores(parallel_configs, train_params)
}

.stop_execution <- function(train_params, model_params, caller) {
  # Check split, n_folds
  void <- all(is.null(train_params$split), is.null(train_params$n_folds))

  if (caller == "classCV") {
    void <- all(void, is.null(model_params$final_model) || isFALSE(model_params$final_model))
  }

  if (void) {
    if (caller == "genFolds") {
      msg <- "neither `train_params$split` or `train_params$n_folds` specified"
    } else {
      msg <- "neither `train_params$split`, `train_params$n_folds`, or `model_params$final_model` specified"
    }
    stop(msg)
  }
}

.check_binary_models <- function(data, formula, target, models, model_params) {
  if (!is.null(formula)) target <- .get_var_names(formula = formula, data = data)$target
  binary_target <- length(levels(factor(data[, target], exclude = NA))) == 2

  obj <- c("reg:logistic", "binary:logistic", "binary:logitraw")
  binary_models <- (any(c("logistic", "regularized_logistic") %in% models) ||
    "xgboost" %in% models && model_params$map_args$xgboost$params$objective %in% obj)

  if (binary_models && !binary_target) {
    stop("'logistic', 'regularized_logistic', and 'xgboost' (with a logistic regression objective) requires a binary target")
  }

  # Check threshold
  if (!is.null(model_params$threshold)) {
    valid_threshold <- model_params$threshold >= 0 && model_params$threshold <= 1
    if (!valid_threshold) stop("`model_params$threshold` must a numeric value from 0 to 1")
  }
}

.check_map_args <- function(model_params, valid_models, error_msg) {
  map_args_models <- names(model_params$map_args)
  if (!is.null(map_args_models) && !all(map_args_models %in% valid_models)) {
    stop(sprintf(error_msg, "model_params$map_args", paste(valid_models, collapse = "', '")))
  }

  if (!is.null(model_params$map_args)) .check_args(model_params = model_params, caller = "model")
}

.check_imputes <- function(valid_imputes, impute_params) {
  msg <- sprintf(
    "invalid method specified in `impute_params$method`, the following is a list of valid methods: '%s'",
    paste(valid_imputes, collapse = "', '")
  )

  if (!is.null(impute_params$method)) {
    if (!impute_params$method %in% valid_imputes) stop(msg)

    if (!is.null(impute_params$args)) .check_args(impute_params = impute_params, caller = "imputation")
  }
}

.check_cores <- function(parallel_configs, train_params) {
  # Check n_cores
  if (!is.null(parallel_configs$n_cores)) {
    if (is.null(train_params$n_folds)) {
      stop("parallel processing is only available when `train_params$n_folds` is not NULL")
    }

    if (parallel_configs$n_cores > as.vector(future::availableCores())) {
      stop(sprintf(
        "more cores specified than available; only %s cores available but %s cores specified",
        as.vector(future::availableCores()), parallel_configs$n_cores
      ))
    }
  }
}
