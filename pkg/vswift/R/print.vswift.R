#' Print parameter information and/or model evaluation metrics
#'
#' @name print
#' @description Prints parameter information and/or model evaluation metrics (classification accuracy, precision,
#'              recall, and f-score for each class) from a vswift object.
#'
#' @param x The vswift object.
#' @param ... Additional arguments to be passed.
#' @param parameters A logical value indicating whether to print parameter information from the vswift object.
#'                   Default = \code{TRUE}.
#' @param metrics A logical value indicating whether to print model evaluation metrics from the vswift object. This will
#'                display the precision, recall, and f-score for each class, along with the means of each metric (if
#'                k-fold validation was used). Default = \code{TRUE}.
#' @param model_type A character or vector of the model information to be printed. If \code{NULL}, all model
#'                   information will be printed. Available options: \code{"lda"} (Linear Discriminant Analysis),
#'                   \code{"qda"} (Quadratic Discriminant Analysis), \code{"logistic"} (Logistic Regression),
#'                   \code{"svm"} (Support Vector Machines), \code{"naivebayes"} (Naive Bayes), \code{"ann"}
#'                   (Artificial Neural Network), \code{"knn"} (K-Nearest Neighbors), \code{"decisiontree"}
#'                   (Decision Tree), \code{"randomforest"} (Random Forest), \code{"multinom"}
#'                   (Multinomial Logistic Regression), \code{"gbm"} (Gradient Boosting Machine). Default = \code{NULL}.
#' @examples
#' # Load an example dataset
#'
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#'
#' result <- classCV(data = iris, target = "Species", split = 0.8,
#' model_type = "lda", stratified = TRUE, random_seed = 123)
#'
#' #  Print parameter information and performance metrics
#' print(result)
#'
#' @author Donisha Smith
#' @export

"print.vswift" <- function(x , ..., parameters = TRUE, metrics = TRUE, model_type = NULL){
  if(inherits(x, "vswift")){
    # List for model names
    model_list <- list("lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis",
                       "svm" = "Support Vector Machines", "ann" = "Neural Network", "decisiontree" = "Decision Tree",
                       "randomforest" = "Random Forest", "gbm" = "Gradient Boosted Machine",
                       "multinom" = "Multinomial Logistic Regression", "logistic" = "Logistic Regression",
                       "knn" = "K-Nearest Neighbors", "naivebayes" = "Naive Bayes")

    # Get models
    if(is.null(model_type)){
      models <- x[["parameters"]][["model_type"]]
    } else {
      # Make lowercase
      model_type <- sapply(model_type, function(x) tolower(x))
      models <- intersect(model_type, x[["parameters"]][["model_type"]])
      if(length(models) == 0){
        stop("no models specified in model_type")
      }
      # Warning when invalid models specified
      invalid_models <- model_type[which(!model_type %in% models)]
      if(length(invalid_models) > 0){
        warning(sprintf("invalid model in model_type or information for specified model not present in vswift x: %s",
                        paste(unlist(invalid_models), collapse = ", ")))
      }
    }

    # Calculate string length of classes
    string_length <- sapply(unlist(x["classes"]), function(x) nchar(x))
    max_string_length <- max(string_length)
    cat("\n")
    partial_output_names <- "Average Precision:  Average Recall:  Average F-score:\n\n"
    cat(rep("-",nchar(paste("Class:",rep("", max_string_length),partial_output_names))[1] %/% 1.5),"\n")
    cat("\n\n")

    for(model in models){
      if(model %in% names(x[["metrics"]])){
        if(parameters == TRUE){
          # Print parameter information
          cat(sprintf("Model: %s\n\n", model_list[[model]]))
          if(model == "logistic"){
            cat(sprintf("Threshold: %s\n\n", x[["parameters"]][["threshold"]]))
          }
          # Creating response variable
          if(length(x[["parameters"]][["predictors"]]) > 20){
            cat(sprintf("Number of Predictors: %s\n\n", length(x[["parameters"]][["predictors"]])))
          } else {
            cat(sprintf("Predictors: %s\n\n", paste(x[["parameters"]][["predictors"]], collapse = ", ")))
          }
          cat(sprintf("Target: %s\n\n", x[["parameters"]][["target"]]))
          cat(sprintf("Formula: %s\n\n", deparse(x[["formula"]])))
          cat(sprintf("Classes: %s\n\n", paste(unlist(x[["classes"]]), collapse = ", ")))
          cat(sprintf("Fold size: %s\n\n", x[["parameters"]][["n_folds"]]))
          cat(sprintf("Split: %s\n\n", x[["parameters"]][["split"]]))
          cat(sprintf("Stratified Sampling: %s\n\n", x[["parameters"]][["stratified"]]))
          cat(sprintf("Random Seed: %s\n\n", x[["parameters"]][["random_seed"]]))
          # Print sample size and missing data for user transparency
          cat(sprintf("Imputation Method: %s\n\n", x[["parameters"]][["impute_method"]]))
          # Print arguments for imputation method
          if(!is.null(x[["parameters"]][["impute_method"]])){
            name_vec <- names(x[["parameters"]][["impute_args"]])
            if(!is.null(x[["parameters"]][["impute_args"]])){
              cat(sprintf("Imputation Arguments: %s\n\n",
                          paste(name_vec, "=", x[["parameters"]][["impute_args"]], collapse = " , ")))
            }
          }
          cat(sprintf("Missing Data: %s\n\n", x[["parameters"]][["missing_data"]]))
          cat(sprintf("Sample Size: %s\n\n", x[["parameters"]][["sample_size"]]))
          # Check for additional arguments
          if(!is.null(x[["parameters"]][["additional_arguments"]])){
            model_names <-  c("lda", "qda", "logistic", "svm", "naivebayes", "ann", "knn", "decisiontree",
                              "randomforest", "multinom", "gbm")
            if(any(model_names %in% names(x[["parameters"]][["additional_arguments"]]))){
              additional_args <- x[["parameters"]][["additional_arguments"]][[model]]
            } else {
              additional_args <- x[["parameters"]][["additional_arguments"]]
            }
            arguments_list <- list()
            for(name in names(additional_args)){
              if(all(name == "params", model == "gbm")){
                param_list <- list()
                for(param in names(additional_args[[name]])){
                  param_list <- c(param_list, paste(param, "=", additional_args[[name]][[param]], collapse = ", "))
                }
                arguments_list <- c(arguments_list, sprintf("%s = %s", name, list(param_list)))
              } else {
                # Add equal sign between name and value
                arguments_list <- c(arguments_list, paste(name, "=", additional_args[[name]], collapse = " "))
              }
            }
            cat(sprintf("Additional Arguments: %s\n\n", paste(unlist(arguments_list), collapse = ", ")))
          }
          # Print information for parallel processing
          cat(sprintf("Parallel: %s\n\n", x[["parameters"]][["parallel"]]))
          if(x[["parameters"]][["parallel"]] == TRUE){
            cat(sprintf("n_cores: %s\n\n", x[["parameters"]][["n_cores"]]))
          }
        }
        # Print metrics
        if(metrics == TRUE){
          # Print model name if parameter is FALSE
          if(parameters == FALSE){
            cat(paste("Model Type:", model_list[[model]]), "\n\n")
          }
          string_diff <- max_string_length - string_length
          # Simplify parameter name
          df <- x[["metrics"]][[model]]
          # Print metrics metrics for train test set if the dataframe exists
          if(is.data.frame(df[["split"]])){
            for(set in c("Training", "Test")){
              # Variable for which class string length to print to ensure all values have equal spacing
              class_position <- 1
              # Print name of the set metrics to be printed and add underscores
              cat("\n\n", set, "\n")
              cat(rep("_", nchar(set)), "\n\n")
              # Print classification accuracy
              cat("Classification Accuracy: ", format(round(df[["split"]][which(df[["split"]]$Set == set), "Classification Accuracy"], 2), nsmall = 2), "\n\n")
              # Print name of metrics
              cat("Class:",rep("", max_string_length),"Precision:  Recall:  F-Score:\n\n")
              # For loop to obtain vector of values for each class
              for(class in unlist(x["classes"])){
                # Empty class_col or initialize variable
                class_col <- c()
                # Go through column names, split the colnames and class name to see if the column name is the metric for that class
                for(colname in colnames(df[["split"]])){
                  split_colname <- unlist(strsplit(colname, split = " "))
                  split_classname <- unlist(strsplit(class, split = " "))
                  if(all(split_classname %in% split_colname)){
                    # Store colnames for the class is variable
                    class_col <- c(class_col, colname)
                  }
                }
                # Print metric corresponding to class
                class_metrics <- sapply(df[["split"]][which(df[["split"]]$Set == set),class_col], function(x) format(round(x,2), nsmall = 2))
                # Add spacing
                padding <- nchar(paste("Class:", rep("", max_string_length),"Pre"))[1]
                if(class_metrics[1] == "NaN"){
                  class_metrics <- c(class_metrics[1], rep("", 5), class_metrics[2],rep("", 5), class_metrics[3])
                } else {
                  class_metrics <- c(class_metrics[1], rep("", 4),class_metrics[2], rep("", 5), class_metrics[3])
                }
                cat(class,rep("", (padding + string_diff[class_position])), paste(class_metrics, collapse = " "), "\n")
                class_position <- class_position + 1
              }
            }
          }
          if(is.data.frame(df[["cv"]])){
            # Variable for which class string length to print to ensure all values have equal spacing
            class_position <- 1
            # Get number of folds to select the correct rows for mean and stdev
            n_folds <- x[["parameters"]][["n_folds"]]
            # Print parameters name
            cat("\n\n", "K-fold CV","\n")
            cat(rep("_",nchar("K-fold CV")), "\n\n")
            mean_cv <- round(df[["cv"]][which(df[["cv"]]$Fold == "Mean CV:"),"Classification Accuracy"],2)
            sd_cv <- round(df[["cv"]][which(df[["cv"]]$Fold == "Standard Deviation CV:"),"Classification Accuracy"],2)
            classification_accuracy_metrics <- c(format(mean_cv, nsmall = 2),
                                                 format(sd_cv, nsmall = 2))

            classification_accuracy_metrics <- sprintf("%s (%s)", classification_accuracy_metrics[1],
                                                       classification_accuracy_metrics[2])
            cat("Average Classification Accuracy: ", classification_accuracy_metrics ,"\n\n")
            cat("Class:",rep("", max_string_length),"Average Precision:  Average Recall:  Average F-score:\n\n")
            # Go through column names, split the colnames and class name to see if the column name is the metric for that class
            for(class in as.character(unlist(x["classes"]))){
              # Empty class_col or initialize variable
              class_col <- c()
              for(colname in colnames(df[["cv"]])){
                split_colname <- unlist(strsplit(colname, split = " "))
                split_classname <- unlist(strsplit(class, split = " "))
                if(all(split_classname %in% split_colname)){
                  # Store colnames for the class is variable
                  class_col <- c(class_col, colname)
                }
              }
              # Print metric corresponding to class
              mean_class_metrics <- sapply(df[["cv"]][((n_folds+1)), class_col], function(x) format(round(x,2), nsmall = 2))
              sd_class_metrics <- sapply(df[["cv"]][((n_folds+2)), class_col], function(x) format(round(x,2), nsmall = 2))
              sd_metric_position <- 1
              class_metrics <- c()
              for(metric in mean_class_metrics){
                class_metrics <- c(class_metrics, sprintf("%s (%s)", metric, sd_class_metrics[sd_metric_position]))
                sd_metric_position <- sd_metric_position + 1
              }
              if(class_metrics[1] == "NaN (NA)"){
                class_metrics <- c(rep("", 3), class_metrics[1],rep("", 6),class_metrics[2],rep("", 6), class_metrics[3])
              } else {
                class_metrics <- c(class_metrics[1], rep("", 6), class_metrics[2], rep("", 6), class_metrics[3])
              }
              # Add spacing
              padding <- nchar(paste("Class:", rep("", max_string_length),"Av"))[1]
              cat(class,rep("",(padding + string_diff[class_position])),paste(class_metrics),"\n")
              # Reset variables
              class_position <- class_position + 1
              sd_metric_position <- 1
              class_metrics <- c()
            }
          }
        }
        # Add space and separation
        cat("\n\n")
        partial_output_names <- "Average Precision:  Average Recall:  Average F-score:\n\n"
        cat(rep("-", nchar(paste("Class:", rep("", max_string_length), partial_output_names))[1] %/% 1.5), "\n")
        cat("\n\n")
      }
    }
  }
}
