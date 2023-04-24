.remove_untrained_observations <- function(trained_data,test_data,response_var,fold = NULL){
  check_predictor_levels <- list()
  for(col in colnames(trained_data[colnames(trained_data) != response_var])){
    if(is.character(trained_data[,col]) | is.factor(trained_data[,col])){
      check_predictor_levels[[col]] <- names(table(trained_data[,col]))[which(as.numeric(table(trained_data[,col])) != 0)]
    }
  }
  
  #Check new columns and set certain predictors in NA if the model has not been trained on
  for(col in colnames(test_data)[colnames(test_data) != response_var]){
    if(is.character(test_data[,col]) | is.factor(test_data[,col])){
      missing <- names(table(test_data[,col]))[which(!(names(table(test_data[,col])) %in% check_predictor_levels[[col]]))]
      if(length(missing) > 0){
        delete_rows <- which(test_data[,col] %in% missing)
        observations <- row.names(test_data)[delete_rows]
        if(is.null(fold)){
          warning(sprintf("for predictor `%s` in test set has at least one class the model has not trained on\n these observations have been removed: %s", col,paste(observations, collapse = ",")))
        } else{
          warning(sprintf("for predictor `%s` in validation set - fold %s has at least one class the model has not trained on\n these observations have been removed: %s", col,fold, paste(observations, collapse = ",")))
        }
        test_data <- test_data[-delete_rows,] 
      }
    }
  }
  return(test_data)
}