# Helper function to store all model information that will be contained in the main output of the classCV function.
#' @importFrom stats as.formula
.store_parameters <- function(formula = NULL,
                              missing_n,
                              preprocessed_data,
                              vars,
                              models,
                              model_params,
                              train_params,
                              impute_params,
                              save,
                              parallel_configs
                              ) {

  
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
    info_dict$configs$train_params <- train_params
    info_dict$configs$missing_data  <- missing_n
    info_dict$configs$effective_sample_size <- nrow(preprocessed_data)
    info_dict$configs$impute_params <- impute_params
    info_dict$configs$parallel_configs <- parallel_configs
    info_dict$configs$save <- save
    
    # Create sublist for class_summary and data_partitions
    info_dict <- c(info_dict, .append_output(preprocessed_data[,vars$target], train_params$stratified))
    # Return output
    return(info_dict)
}
