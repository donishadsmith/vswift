# Helper function to store all model information that will be contained in the main output of the classCV function.
#' @noRd
#' @export
.store_parameters <- function(formula = NULL, data = NULL, preprocessed_data = NULL, predictor_vec = NULL,
                              target = NULL, model_type = NULL, threshold = NULL, split = NULL, n_folds = NULL,
                              stratified = NULL, random_seed = NULL, mod_args = NULL, n_cores = NULL, parallel = NULL,
                              impute_method = NULL, impute_args = NULL, classCV_output = NULL, ...){
  if(is.null(impute_method)){
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
    store_parameters_output[["parameters"]][["parallel"]] <- ifelse(!is.null(parallel), TRUE, FALSE)
    store_parameters_output[["parameters"]][["n_cores"]] <- n_cores

    if(!is.null(mod_args)){
      store_parameters_output[["parameters"]][["additional_arguments"]] <- mod_args
    } else {
      store_parameters_output[["parameters"]][["additional_arguments"]] <- list(...)
    }

    # Store classes
    store_parameters_output[["classes"]][[target]] <- names(table(factor(preprocessed_data[,target])))
    # Create formula string
    if(is.null(formula)){
      store_parameters_output[["formula"]] <- as.formula(paste(target, "~", paste(predictor_vec, collapse = " + ")))
    } else{
      store_parameters_output[["formula"]] <- formula
    }

    # Return output
    return(store_parameters_output)
  } else{
    classCV_output[["parameters"]][["impute_method"]] <- impute_method
    classCV_output[["parameters"]][["impute_args"]] <- impute_args
    # Return output
    return(classCV_output)
  }
}
