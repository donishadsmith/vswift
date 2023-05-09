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
#' @export
"plot.vswift" <- function(object, split = TRUE, cv = TRUE, save_plots = FALSE, path = NULL, ...){
  if(class(object) == "vswift"){
    if(save_plots == FALSE){
      vswift:::.visible_plots(object = object, split = split, cv = cv)
    }else{
      vswift:::.save_plots(object = object, split = split, cv = cv, path = path, ...)
    }
  }else{
    stop("object must be of class 'vswift'")
  }
}
