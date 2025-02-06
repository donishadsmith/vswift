#' Plot Model Evaluation Metrics
#'
#' @aliases plot.vswift
#'
#' @description Plots classification metrics (accuracy, precision, recall, and f1 for each class).
#'
#' @param x A list object of class \code{"vswift"}.
#'
#' @param metrics A character vector indicating which metrics to plot. Supported options are \code{"accuracy"},
#' \code{"precision"}, \code{"recall"}, and \code{"f1"}. Default is \code{c("accuracy", "precision", "recall", "f1")}.

#'
#' @param models A character string or a character vector specifying the classification algorithm(s) evaluation metrics
#' to plot. If \code{NULL}, all models will be plotted. The following options are available:
#' \itemize{
#'  \item \code{"lda"}: Linear Discriminant Analysis
#'  \item \code{"qda"}: Quadratic Discriminant Analysis
#'  \item \code{"logistic"}: Unregularized Logistic Regression
#'  \item \code{"regularized_logistic"}: Regularized Logistic Regression
#'  \item \code{"svm"}: Support Vector Machine
#'  \item \code{"naivebayes"}: Naive Bayes
#'  \item \code{"nnet"}: Neural Network
#'  \item \code{"knn"}: K-Nearest Neighbors
#'  \item \code{"decisiontree"}: Decision Tree
#'  \item \code{"randomforest"}: Random Forest
#'  \item \code{"multinom"}: Unregularized Multinomial Logistic Regression
#'  \item \code{"regularized_multinomial"}: Regularized Multinomial Logistic Regression
#'  \item \code{"xgboost"}: Extreme Gradient Boosting
#'  }
#'  Default = \code{NULL}.
#'
#' @param split A logical value indicating whether to plot metrics for the train-test split results. Default is
#' \code{TRUE}.
#'
#' @param cv A logical value indicating whether to plot metrics for cross-validation results. Default is \code{TRUE}.
#'
#' @param class_names A vector of the specific classes to plot. If \code{NULL}, plots are generated for all classes.
#' Default is \code{NULL}.
#'
#' @param path A character string specifying the directory (with a trailing slash) to save the plots.
#' Default is \code{NULL}.
#'
#' @param ... Additional arguments passed to the \code{png} function.
#'
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set and stratified sampling using QDA
#'
#' result <- classCV(
#'   data = iris,
#'   target = "Species",
#'   models = "qda",
#'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 123),
#'   save = list(models = TRUE)
#' )
#'
#'
#' # Plot performance metrics for train-test split
#'
#' plot(result, class_names = "setosa", metrics = "f1")
#'
#' @importFrom grDevices dev.off dev.new graphics.off png
#' @importFrom  graphics axis abline legend
#'
#' @author Donisha Smith
#' @method plot vswift
#'
#' @export
"plot.vswift" <- function(x, metrics = c("accuracy", "precision", "recall", "f1"), models = NULL, split = TRUE,
                          cv = TRUE, class_names = NULL, path = NULL, ...) {
  # Lowercase and intersect common names
  metrics <- intersect(unlist(lapply(metrics, function(x) tolower(x))), c("accuracy", "precision", "recall", "f1"))
  if (length(metrics) == 0) {
    stop(sprintf("no metrics specified, available metrics: %s", paste(c("accuracy", "precision", "recall", "f1"), collapse = ", ")))
  }
  # intersect common names
  if (!is.null(class_names)) {
    class_names <- intersect(class_names, x$class_summary$classes)
    if (length(class_names) == 0) {
      stop(sprintf("no classes specified, available classes: %s", paste(x$class_summary$classes, collapse = ", ")))
    }
  }

  # Get models
  models <- .intersect_models(x, models)

  # Iterate over models
  for (model in models) {
    .plot(
      x = x, metrics = metrics, model = model, plot_title = .MODEL_LIST[[model]], split = split, cv = cv,
      class_names = class_names, path = path, ...
    )
  }
}
