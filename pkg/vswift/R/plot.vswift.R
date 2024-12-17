#' Plot model evaluation metrics
#'
#' @name plot
#' @description Plots model evaluation metrics (classification accuracy and precision, recall, and f-score for each
#'              class) from a vswift object.
#'
#' @param x An vswift object.
#' @param split A logical value indicating whether to plot metrics for train-test splitting results.
#'              Default = \code{TRUE}.
#' @param cv A logical value indicating whether to plot metrics for cross-validation results.
#'           Note: Solid red line represents the mean and dashed blue line represents the standard deviation.
#'           Default = \code{TRUE}.
#' @param metrics A vector consisting of which metrics to plot. Available metrics includes, \code{"accuracy"},
#'                \code{"precision"}, \code{"recall"}, \code{"f1"}.
#'                Default = \code{c("accuracy","precision", "recall", "f1")}.
#' @param class_names A vector consisting of class names to plot. If NULL, plots are generated for each class.
#'                    Default = \code{NULL}.
#' @param save_plots A logical value to save all plots as separate png files. Plot will not be displayed if set to TRUE.
#'                   Default = \code{FALSE}.
#' @param path A character representing the file location, with trailing slash, to save to. If not specified, the plots
#'             will be saved to the current working directory. Default = \code{NULL}.
#' @param models A character or vector of the model metrics to be printed. If \code{NULL}, all model metrics will
#'                   be printed. Available options: \code{"lda"} (Linear Discriminant Analysis), \code{"qda"}
#'                   (Quadratic Discriminant Analysis), \code{"logistic"} (unregularized Logistic Regression), \code{"svm"}
#'                   (Support Vector Machine), \code{"naivebayes"} (Naive Bayes), \code{"nnet"}
#'                   (Neural Network), \code{"knn"} (K-Nearest Neighbors), \code{"decisiontree"}
#'                   (Decision Tree), \code{"randomforest"} (Random Forest), \code{"multinomial"}
#'                   (unregularized Multinomial Logistic Regression), \code{"xgboost"} (Extreme Gradient Boosting).
#'                   Default = \code{NULL}.
#' @param ... Additional arguments that can be passed to the \code{png} function.
#'
#' @return Plots representing evaluation metrics.
#' @examples
#' # Load an example dataset
#'
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set and stratified_sampling using QDA
#'
#' result <- classCV(
#'   data = iris,
#'   target = "Species",
#'   models = "qda",
#'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 50)
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
#' @export

"plot.vswift" <- function(x, ..., split = TRUE, cv = TRUE, metrics = c("accuracy", "precision", "recall", "f1"),
                          class_names = NULL, save_plots = FALSE, path = NULL, models = NULL) {
  if (inherits(x, "vswift")) {
    # Create list
    model_list <- list(
      "lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis",
      "svm" = "Support Vector Machine", "nnet" = "Neural Network", "decisiontree" = "Decision Tree",
      "randomforest" = "Random Forest", "xgboost" = "Extreme Gradient Boosting",
      "multinom" = "Multinomial Logistic Regression", "logistic" = "Logistic Regression",
      "knn" = "K-Nearest Neighbors", "naivebayes" = "Naive Bayes"
    )

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
    if (is.null(models)) {
      models <- x$configs$models
    } else {
      # Make lowercase
      models <- sapply(models, function(x) tolower(x))
      models <- intersect(models, x$configs$models)
      if (length(models) == 0) stop("no valid models specified in `models`")

      # Warning when invalid models specified
      invalid_models <- models[which(!models %in% models)]
      if (length(invalid_models) > 0) {
        warning(sprintf(
          "invalid model in models or information for specified model not present in vswift x: %s",
          paste(unlist(invalid_models), collapse = ", ")
        ))
      }
    }

    # Iterate over models
    for (model in models) {
      if (save_plots == FALSE) {
        .visible_plots(
          x = x, split = split, cv = cv, metrics = metrics, class_names = class_names, model_name = model,
          model_list = model_list
        )
      } else {
        .save_plots(
          x = x, split = split, cv = cv, metrics = metrics, class_names = class_names, path = path,
          model_name = model, model_list = model_list, ...
        )
      }
    }
  }
}
