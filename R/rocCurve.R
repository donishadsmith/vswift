#' Plot Receiver Operating Characteristic (ROC) Curves for Binary Classification Tasks
#'
#' @name rocCurve
#'
#' @description Produces ROC curves and computes the area under the curve (AUC) and Youdin's Index.
#' Only works for binary classification tasks.
#'
#' @param x A list object of class \code{"vswift"}. Note that the models must be saved using
#' \code{save = list("models" = TRUE)} in \code{classCV} for this function to work.
#'
#' @param data A data frame. If \code{NULL}, then the preprocessed data muse be saved using
#' \code{save = list("data" = TRUE)} in \code{classCV} Default = \code{NULL}.
#'
#' @param models A character string or a character vector specifying the classification algorithm(s) to plot ROC curves
#' for. If \code{NULL}, all models will be plotted. The following options are available:
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
#' @param split A logical value indicating whether to plot ROC curves for the train-test split results. Default is
#' \code{TRUE}.
#'
#' @param cv A logical value indicating whether to plot ROC curves for cross-validation results. Default is \code{TRUE}.
#'
#' @param thresholds A numerical vector specifying the thresholds to use when producing the ROC curves. If left as NULL
#' the unique probability values produced by the training model will be used as thresholds. Default is \code{NULL}.
#'
#' @param return_output A logical value indicating whether to return the output list. Default is \code{TRUE}.
#'
#' @param path A character string specifying the directory (with a trailing slash) to save the plots.
#' Default is \code{NULL}.
#'
#' @param ... Additional arguments passed to the \code{png} function.
#'
#' @return A list containing thresholds used to generate the ROC, target labels, False Positive Rates (FPR),
#' True Positive Rates (TPR), Area Under The Curve (AUC), and Youdin's Index for all training and validation sets
#' for each model
#'
#' @examples
#' # Load an example dataset
#' data <- iris
#'
#' # Make Binary
#' data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
#'
#' # Perform a train-test split with an 80% training set and stratified sampling using QDA
#' result <- classCV(
#'   data = data,
#'   target = "Species",
#'   models = "qda",
#'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 123),
#'   save = list(data = TRUE, models = TRUE)
#' )
#'
#' # Get ROC curve
#' rocCurve(result, return_output = FALSE)
#'
#' @author Donisha Smith
#'
#' @importFrom grDevices rainbow
#' @importFrom graphics lines
#'
#' @export
rocCurve <- function(x, data = NULL, models = NULL, split = TRUE, cv = TRUE, thresholds = NULL, return_output = TRUE,
                     path = NULL, ...) {
  if (inherits(x, "vswift")) {
    # Perform checks and get dictionary class keys and variables
    info <- .perform_checks(x, data)

    # Unlist keys to turn into a named vector
    info$keys <- unlist(info$keys)

    # Get valid models
    models <- .intersect_models(x, models)

    if ("xgboost" %in% models && x$configs$model_params$map_args$xgboost$params$objective == "multi:softmax") {
      warnings("'xgboost' cannot be specified when the 'multi:softmax; objective is used since probabilties are needed")
      models <- models[!models == "xgboost"]
    }

    if (length(models) == 0) stop("no valid models to plot")

    # Iterate over models
    roc_output <- list()

    for (model in models) {
      roc_output[[model]] <- .computeROC(
        x, data, model, .MODEL_LIST[[model]], split, cv, thresholds, info, path, ...
      )

      if (!isTRUE(return_output)) roc_output[[model]] <- NULL
    }

    if (isTRUE(return_output)) {
      return(roc_output)
    }
  } else {
    stop("`x` must be an object of class 'vswift'")
  }
}
