.store_parameters <- function(data = NULL, preprocessed_data = NULL, predictor_vec = NULL, target = NULL, model_type = NULL,
                               threshold = NULL, split = NULL, n_folds = NULL, stratified = NULL, random_seed = NULL, mod_args = NULL, n_cores = NULL, parallel = FALSE, ...){
  # Initialize output list
  store_parameters_output <- list()
  store_parameters_output[["analysis_type"]] <- "classification"
  store_parameters_output[["parameters"]] <- list()
  store_parameters_output[["parameters"]][["predictors"]] <- predictor_vec
  store_parameters_output[["parameters"]][["target"]]  <- target
  store_parameters_output[["parameters"]][["model_type"]] <- model_type
  if("logistic" %in% model_type) store_parameters_output[["parameters"]][["threshold"]] <- threshold
  store_parameters_output[["parameters"]][["split"]]  <- split
  store_parameters_output[["parameters"]][["n_folds"]]  <- n_folds
  store_parameters_output[["parameters"]][["stratified"]]  <- stratified
  store_parameters_output[["parameters"]][["random_seed"]]  <- random_seed
  store_parameters_output[["parameters"]][["missing_data"]]  <- nrow(data) - nrow(preprocessed_data)
  store_parameters_output[["parameters"]][["sample_size"]] <- nrow(preprocessed_data)
  store_parameters_output[["parameters"]][["parallel"]] <- parallel
  store_parameters_output[["parameters"]][["n_cores"]] <- n_cores
  
  if(!is.null(mod_args)){
    store_parameters_output[["parameters"]][["additional_arguments"]] <- mod_args
  } else {
    store_parameters_output[["parameters"]][["additional_arguments"]] <- list(...)
  }
  
  # Store classes
  store_parameters_output[["classes"]][[target]] <- names(table(factor(preprocessed_data[,target])))
  # Create formula string
  store_parameters_output[["formula"]] <- as.formula(paste(target, "~", paste(predictor_vec, collapse = " + ")))
  # Return output
  return(store_parameters_output)
}