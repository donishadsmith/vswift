# Function to expand the dataframe containing the performance metrics
#' @noRd
#' @export
.expand_dataframe <- function(classCV_output, split, n_folds, model_type){
  if(any(!is.null(split), !is.null(n_folds))){
    target <- classCV_output[["parameters"]][["target"]]
    column_names <- c("Classification Accuracy",lapply(classCV_output[["classes"]][[target]], function(x) paste(paste("Class:", x), c("Precision", "Recall", "F-Score"))))
    if(!is.null(split)){
      split_df <- data.frame(sapply(unlist(column_names), function(x) assign(x, rep(NA,2))))
      colnames(split_df) <- unlist(column_names)
      for(model_name in model_type){
        classCV_output[["metrics"]][[model_name]][["split"]] <- cbind(data.frame("Set" = c("Training", "Test")), split_df)
      }
    }
    if(!is.null(n_folds)){
      cv_df <- data.frame(sapply(unlist(column_names), function(x) assign(x, rep(NA,n_folds))))
      colnames(cv_df) <- unlist(column_names)
      for(model_name in model_type){
        classCV_output[["metrics"]][[model_name]][["cv"]] <- cbind(data.frame("Fold" = paste("Fold", 1:n_folds)),cv_df)
      }
    }
    return(classCV_output)
  }
}
