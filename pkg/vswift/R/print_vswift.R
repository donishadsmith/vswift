#' Print parameter information and/or model evaluation metrics
#'
#' `print.vswift` prints parameter information and/or model evaluation metrics (classification accuracy and precision, recall, and f-score for each class) from a vswift object. 
#' 
#' 
#' @param object An object of class vswift.
#' @param parameters A logical value indicating whether to print parameter information from the vswift object. Default is set to TRUE.
#' @param metrics A logical value indicating whether to print model evaluation metrics from the vswift object. This will display the precision, recall, and f-score for each class.
#' If the vswift object contains information for train-test splitting, the classification accuracy for the training and test set as well as the precision, recall, and f-score for each class
#' will be displayed. If the vswift object contains information for k-fold validation, the mean and standard deviation for the classification accuracy and class' precision, recall, and f-score will be displayed. Default is set to TRUE.
#'
#' @examples
#' # Load an example dataset
#' 
#' data(iris)
#' 
#' # Perform a train-test split with an 80% training set using LDA
#'
#' result <- categorical_cv_split(data = iris, target = "Species", split = 0.8,
#' model_type = "lda", stratified = TRUE, random_seed = 123)
#' 
#' #  Print parameter information and performance metrics
#' print(result)
#'
#' @export
"print.vswift"<- function(object, parameters = TRUE, metrics = TRUE){
  if(class(object) == "vswift"){
    if(parameters == TRUE){
      # Print parameter information
      cat(sprintf("Model Type: %s\n\n", object[["information"]][["parameters"]][["model_type"]]))
      if(object[["information"]][["parameters"]][["model_type"]] == "logistic"){
        cat(sprintf("Threshold: %s\n\n", object[["information"]][["parameters"]][["threshold"]]))
      }
      # Creating response variable
      cat(sprintf("Predictors: %s\n\n", paste(object[["information"]][["parameters"]][["predictors"]], collapse = ",")))
      cat(sprintf("Target: %s\n\n", object[["information"]][["parameters"]][["responsd_variable"]]))
      cat(sprintf("Classes: %s\n\n", paste(unlist(object[["classes"]]), collapse = ", ")))
      cat(sprintf("Fold size: %s\n\n", object[["information"]][["parameters"]][["n_folds"]]))
      cat(sprintf("Split: %s\n\n", object[["information"]][["parameters"]][["split"]]))
      cat(sprintf("Stratified Sampling: %s\n\n", object[["information"]][["parameters"]][["stratified"]]))
      cat(sprintf("Random Seed: %s\n\n", object[["information"]][["parameters"]][["random_seed"]]))
      # Print sample size and missing data for user transparency
      cat(sprintf("Missing Data: %s\n\n", object[["information"]][["parameters"]][["missing_data"]]))
      cat(sprintf("Sample Size: %s\n\n", object[["information"]][["parameters"]][["sample_size"]]))
      # Check for additional arguments
      if(!is.null(object[["information"]][["parameters"]][["additional_arguments"]])){
        arguments_list <- list()
        for(name in names(object[["information"]][["parameters"]][["additional_arguments"]])){
          # Add equal sign between name and value
          arguments_list <- c(arguments_list,paste(name, "=",object[["information"]][["parameters"]][["additional_arguments"]][[name]], collapse = " "))
        }
        cat(sprintf("Additional Arguments: %s\n\n", paste(unlist(arguments_list), collapse = ", ")))
      }
    }
  }
  # Print metrics
  if(metrics == TRUE){
    # Calculate string length of classes
    string_length <- sapply(unlist(object["classes"]), function(x) nchar(x))
    max_string_length <- max(string_length)
    string_diff <- max_string_length - string_length 
    # Print metrics metrics for train test set if the dataframe exists
    if(is.data.frame(object[["metrics"]][["split"]])){
      for(set in c("Training","Test")){
        # Variable for which class string length to print to ensure all values have equal spacing
        class_position <- 1 
        # Print name of the set metrics to be printed and add underscores
        cat("\n\n",set,"\n")
        cat(rep("_",nchar(set)),"\n\n")
        # Print classification accuracy
        cat("Classication Accuracy: ", format(round(object[["metrics"]][["split"]][which(object[["metrics"]][["split"]]$Set == set),"Classification Accuracy"],2), nsmall = 2),"\n\n")
        # Print name of metrics
        cat("Class:",rep("", max_string_length),"Precision:  Recall:  F-Score:\n\n")
        # For loop to obtain vector of values for each class
        for(class in unlist(object["classes"])){
          # Empty class_col or initialize variable
          class_col <- c()
          # Go through column names, split the colnames and class name to see if the column name is the metric for that class
          for(colname in colnames(object[["metrics"]][["split"]])){
            split_colname <- unlist(strsplit(colname,split = " "))
            split_classname <- unlist(strsplit(class,split = " ")) 
            if(all(split_classname %in% split_colname)){
              # Store colnames for the class is variable
              class_col <- c(class_col, colname)
            }
          }
          # Print metric corresponding to class
          class_metrics <- sapply(object[["metrics"]][["split"]][which(object[["metrics"]][["split"]]$Set == set),class_col], function(x) format(round(x,2), nsmall = 2))  
          # Add spacing
          padding <- nchar(paste("Class:",rep("", max_string_length),"Pre"))[1]
          if(class_metrics[1] == "NaN"){
            class_metrics <- c(class_metrics[1],rep("", 5),class_metrics[2],rep("", 5),class_metrics[3])
          }else{
            class_metrics <- c(class_metrics[1],rep("", 4),class_metrics[2],rep("", 5),class_metrics[3])
          }
          cat(class,rep("",(padding + string_diff[class_position])),paste(class_metrics, collapse = " "),"\n")
          class_position <- class_position + 1
        }
      }
    }
    if(is.data.frame(object[["metrics"]][["cv"]])){
      # Variable for which class string length to print to ensure all values have equal spacing
      class_position <- 1 
      # Get number of folds to select the correct rows for mean and stdev
      n_folds <- object[["information"]][["parameters"]][["n_folds"]]
      # Print parameters name
      cat("\n\n","K-fold CV","\n")
      cat(rep("_",nchar("K-fold CV")),"\n\n")
      classification_accuracy_metrics <- c(format(round(object[["metrics"]][["cv"]][which(object[["metrics"]][["cv"]]$Fold == "Mean CV:"),"Classification Accuracy"],2), nsmall = 2),
                                         format(round(object[["metrics"]][["cv"]][which(object[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),"Classification Accuracy"],2), nsmall = 2))
      
      classification_accuracy_metrics <- sprintf("%s (%s)", classification_accuracy_metrics[1],classification_accuracy_metrics[2])
      cat("Average Classication Accuracy: ", classification_accuracy_metrics ,"\n\n")
      # cat("Class:",rep("", max_string_length),"Average Precision:  StDev Precision:  Average Recall:  StDev Recall:  Average F-score:  StDev F-score:\n\n")
      cat("Class:",rep("", max_string_length),"Average Precision:  Average Recall:  Average F-score:\n\n")
      # Go through column names, split the colnames and class name to see if the column name is the metric for that class
      for(class in as.character(unlist(object["classes"]))){
        # Empty class_col or initialize variable
        class_col <- c()
        for(colname in colnames(object[["metrics"]][["cv"]])){
          split_colname <- unlist(strsplit(colname,split = " "))
          split_classname <- unlist(strsplit(class,split = " ")) 
          if(all(split_classname %in% split_colname)){
            # Store colnames for the class is variable
            class_col <- c(class_col, colname)
          }
        }
        # Print metric corresponding to class
        mean_class_metrics <- sapply(object[["metrics"]][["cv"]][((n_folds+1)),class_col], function(x) format(round(x,2), nsmall = 2))  
        sd_class_metrics <- sapply(object[["metrics"]][["cv"]][((n_folds+2)),class_col], function(x) format(round(x,2), nsmall = 2))  
        sd_metric_position <- 1
        class_metrics <- c()
        for(metric in mean_class_metrics){
          class_metrics <- c(class_metrics, sprintf("%s (%s)", metric, sd_class_metrics[sd_metric_position]))
          sd_metric_position <- sd_metric_position + 1
        }
        if(class_metrics[1] == "NaN (NA)"){
          class_metrics <- c(rep("", 3),class_metrics[1],rep("", 6),class_metrics[2],rep("", 6), class_metrics[3])
        }else{
          class_metrics <- c(class_metrics[1],rep("", 6),class_metrics[2],rep("", 6), class_metrics[3])
        }
        # Add spacing
        padding <- nchar(paste("Class:",rep("", max_string_length),"Av"))[1]
        cat(class,rep("",(padding + string_diff[class_position])),paste(class_metrics),"\n")
        # Reset variables
        class_position <- class_position + 1
        sd_metric_position <- 1
        class_metrics <- c()
      }
    }
  }
}
