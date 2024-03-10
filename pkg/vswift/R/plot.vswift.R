#' Plot model evaluation metrics
#' 
#' `plot.vswift` plots model evaluation metrics (classification accuracy and precision, recall, and f-score for each class) from a vswift x. 
#'
#'
#' @param x An x of class vswift.
#' @param split A logical value indicating whether to plot metrics for train-test splitting results. Default = TRUE.
#' @param cv A logical value indicating whether to plot metrics for k-fold cross-validation results. Note: Solid red line represents the mean
#' and dashed blue line represents the standard deviation. Default = TRUE.
#' @param metrics A vector consisting of which metrics to plot. Available metrics includes, "accuracy", "precision", "recall", "f1".
#' Default = c("accuracy","precision", "recall", "f1").
#' @param class_names A vector consisting of class names to plot. If NULL, plots are generated for each class.Defaeult = NULL
#' @param save_plots A logical value to save all plots as separate png files. Plot will not be displayed if set to TRUE. Default = FALSE.
#' @param path A character representing the file location, with trailing slash, to save to. If not specified, the plots will be saved to the current
#' working directory.
#' @param model_type A character or vector of the model metrics to be printed. If not specified, all model metrics will be printed. Available options:
#'                   "lda" (Linear Discriminant Analysis), "qda" (Quadratic Discriminant Analysis), 
#'                   "logistic" (Logistic Regression), "svm" (Support Vector Machines), "naivebayes" (Naive Bayes), 
#'                   "ann" (Artificial Neural Network), "knn" (K-Nearest Neighbors), "decisiontree" (Decision Tree), 
#'                   "randomforest" (Random Forest), "multinom" (Multinomial Logistic Regression), "gbm" (Gradient Boosting Machine).
#' @param ... Additional arguments that can be passed to the `png()` function.
#' 
#' 
#' @return Plots representing evaluation metrics.
#' @examples
#' # Load an example dataset
#' 
#' data(iris)
#' 
#' # Perform a train-test split with an 80% training set and stratified_sampling using QDA
#' 
#' result <- classCV(data = iris, target = "Species", split = 0.8,
#' model_type = "qda", stratified = TRUE)
#' 
#' # Plot performance metrics for train-test split
#' 
#' plot(result, class_names = "setosa", metrics = "f1")
#' 
#' @author Donisha Smith
#' @importFrom grDevices dev.new dev.off graphics.off png
#' @importFrom graphics abline axis
#' @export

"plot.vswift" <- function(x, ..., split = TRUE, cv = TRUE, metrics = c("accuracy","precision", "recall", "f1"), class_names = NULL, save_plots = FALSE, path = NULL, model_type = NULL){
  
  if(inherits(x, "vswift")){
    # Create list
    model_list = list("lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis", "svm" = "Support Vector Machines",
                      "ann" = "Neural Network", "decisiontree" = "Decision Tree", "randomforest" = "Random Forest", "gbm" = "Gradient Boosted Machine",
                      "multinom" = "Multinomial Logistic Regression", "logistic" = "Logistic Regression", "knn" = "K-Nearest Neighbors",
                      "naivebayes" = "Naive Bayes")
    
    # Lowercase and intersect common names
    metrics <- intersect(unlist(lapply(metrics, function(x) tolower(x))), c("accuracy","precision", "recall", "f1"))
    if(length(metrics) == 0){
      stop(sprintf("no metrics specified, available metrics: %s", paste(c("accuracy","precision","recall","f1"), collapse = ", ")))
    }
    # intersect common names
    if(!is.null(class_names)){
      class_names <- intersect(class_names, x[["classes"]][[1]])
      if(length(class_names) == 0){
        stop(sprintf("no classes specified, available classes: %s", paste(x[["classes"]][[1]], collapse = ", ")))
      }
    }
    
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
    # Iterate over models
    for(model in models){
      if(save_plots == FALSE){
        .visible_plots(x = x, split = split, cv = cv, metrics = metrics, class_names = class_names, model_name = model, model_list = model_list)
      } else {
        .save_plots(x = x, split = split, cv = cv, metrics = metrics, class_names = class_names, path = path, model_name = model, model_list = model_list,...)
      }
    }
  } else {
    stop("x must be of class 'vswift'")
  }
}