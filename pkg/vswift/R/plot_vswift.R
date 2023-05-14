#' Plot model evaluation metrics
#' 
#' `plot.vswift` plots model evaluation metrics (classification accuracy and precision, recall, and f-score for each class) from a vswift object. 
#'
#'
#' @param object An object of class vswift.
#' @param split A logical value indicating whether to plot metrics for train-test splitting results. Default = TRUE.
#' @param cv A logical value indicating whether to plot metrics for k-fold cross-validation results. Note: Solid red line represents the mean
#' and dashed blue line represents the standard deviation. Default = TRUE.
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
#' plot(result)
#' 
#' @author Donisha Smith
#' @export

"plot.vswift" <- function(object, split = TRUE, cv = TRUE, save_plots = FALSE, path = NULL, model_type = NULL,...){
  # Create list
  
  model_list = list("lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis", "svm" = "Support Vector Machines",
                    "ann" = "Neural Network", "decisiontree" = "Decision Tree", "randomforest" = "Random Forest", "gbm" = "Gradient Boosted Machine",
                    "multinom" = "Multinomial Logistic Regression", "logistic" = "Logistic Regression", "knn" = "K-Nearest Neighbors",
                    "naivebayes" = "Naive Bayes")
  
  # Get models
  if(is.null(model_type)){
    models <- object[["parameters"]][["model_type"]]
  } else {
    models <- intersect(model_type, object[["parameters"]][["model_type"]])
  }
  
  # Check model_type
  if(length(models) == 0){
    stop("no models specified in model_type")
  }
  if(!all(models %in% names(model_list))){
    stop("invalid model in model_type")
  }
  
  # Iterate over models
  for(model in models){
    if(class(object) == "vswift"){
      if(save_plots == FALSE){
        vswift:::.visible_plots(object = object, split = split, cv = cv, model_name = model, model_list = model_list)
      } else {
        vswift:::.save_plots(object = object, split = split, cv = cv, path = path, model_name = model, model_list = model_list,...)
      }
    } else {
      stop("object must be of class 'vswift'")
    }
  }
}