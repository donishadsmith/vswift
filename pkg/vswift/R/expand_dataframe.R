# Function to expand the dataframe containing the performance metrics
.expand_dataframe <- function(train_params, models, classes) {
  # Create df_list
  df_list <- list()
  # Create base column names
  col_names <- c("Classification Accuracy",
                 sapply(classes, function(x) paste(paste("Class:", x),c("Precision", "Recall", "F-Score"))))
  
  # Create split df
  if (!is.null(train_params$split)) {
    # Create dataframe by using assign combined with rep and sapply to dynamically create columns, each element is NA
    split_df <- data.frame(sapply(col_names, function(x) assign(x, rep(NA, 2))))
    # Fix names
    colnames(split_df) <- col_names
    for (model in models) {
      df_list[[model]]$split <- cbind(data.frame("Set" = c("Training", "Test")), split_df)
    }
  }
  
  # Create cv df
  if (!is.null(train_params$n_folds)) {
    cv_df <- data.frame(sapply(unlist(col_names), function(x) assign(x, rep(NA, train_params$n_folds))))
    colnames(cv_df) <- col_names
    for (model in models) {
      df_list[[model]]$cv <- cbind(data.frame("Fold" = paste("Fold", 1:train_params$n_folds)),cv_df)
    }
  }
    
  return(df_list)
}
