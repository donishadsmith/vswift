# Helper function to store all model information that will be contained in the main output of the classCV function.
.store_parameters <- function(formula = NULL,
                              missing_n,
                              preprocessed_data,
                              vars,
                              models,
                              model_params,
                              train_params,
                              impute_params,
                              save,
                              parallel_configs) {
  # Initialize output list
  info_dict <- list()
  info_dict$configs <- list()
  if (!is.null(formula)) {
    info_dict$configs$formula <- formula
  } else {
    info_dict$configs$formula <- as.formula(paste(vars$target, "~", paste(vars$predictors, collapse = " + ")))
  }
  info_dict$configs$n_features <- length(vars$predictors)
  info_dict$configs$models <- models
  info_dict$configs$model_params <- model_params

  if (!is.null(impute_params$method) && !isTRUE(train_params$standardize)) {
    warning("`train_params$standardize` will be set to TRUE since imputation has been requested")
    train_params$standardize <- TRUE
  }

  info_dict$configs$train_params <- train_params
  info_dict$missing_data_summary$unlabeled_observations <- length(missing_n$unlabeled_data_indices)
  info_dict$missing_data_summary$observations_missing_all_features <- length(missing_n$missing_all_features_indices)
  info_dict$missing_data_summary$incomplete_labeled_observations <- missing_n$n_incomplete_labeled_data
  info_dict$missing_data_summary$complete_observations <- sum(complete.cases(preprocessed_data))
  info_dict$configs$impute_params <- impute_params
  info_dict$configs$parallel_configs <- parallel_configs
  info_dict$configs$save <- save

  # Create sublist for class_summary and data_partitions
  info_dict <- c(info_dict, .append_output(preprocessed_data[, vars$target], train_params$stratified))
  # Return output
  return(info_dict)
}
