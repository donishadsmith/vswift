#' Print parameter information and/or model evaluation metrics
#'
#' @name print
#' @description Prints model configuration information and/or model evaluation metrics (classification accuracy,
#' precision, recall, and f-score for each class) from a vswift object.
#'
#' @param x The vswift object.
#' @param ... Additional arguments to be passed.
#' @param configs A logical value indicating whether to print model configeration information from the vswift object.
#'                   Default = \code{TRUE}.
#' @param metrics A logical value indicating whether to print model evaluation metrics from the vswift object. This will
#'                display the precision, recall, and f-score for each class, along with the means of each metric (if
#'                k-fold validation was used). Default = \code{TRUE}.
#' @param models A character or vector of the model information to be printed. If \code{NULL}, all model
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
#' result <- classCV(
#'   data = iris,
#'   target = "Species",
#'   models = "lda",
#'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 50)
#' )
#'
#' #  Print parameter information and performance metrics
#' print(result)
#'
#' @author Donisha Smith
#' @export

"print.vswift" <- function(x, ..., configs = TRUE, metrics = TRUE, models = NULL) {
  if (inherits(x, "vswift")) {
    # List for model names
    model_list <- list(
      "lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis",
      "svm" = "Support Vector Machines", "ann" = "Neural Network", "decisiontree" = "Decision Tree",
      "randomforest" = "Random Forest", "gbm" = "Gradient Boosted Machine",
      "multinom" = "Multinomial Logistic Regression", "logistic" = "Logistic Regression",
      "knn" = "K-Nearest Neighbors", "naivebayes" = "Naive Bayes"
    )

    # Get models
    if (is.null(models)) {
      models <- x$configs$models
    } else {
      models <- intersect(models, names(x$metrics))
    }

    if (length(models) == 0) stop("no models specified in `models` or `specified` models not found in `x$metrics`")

    # Calculate string length of classes
    str_list <- .dashed_lines(x$class_summary$classes, TRUE)
    for (model in models) {
      cat(paste("Model:", model_list[[model]]), "\n\n")
      # Print parameter information
      if (configs) .print_configs(x, model)

      if (metrics) {
        if (is.data.frame(x$metrics[[model]]$split)) .print_metrics_split(x, x$metrics[[model]]$split, str_list$max, str_list$diff)
        if (is.data.frame(x$metrics[[model]]$cv)) .print_metrics_cv(x, x$metrics[[model]]$cv, str_list$max, str_list$diff)
      }
      # Add dashed line to separate each model
      .dashed_lines(x$class_summary$classes)
    }
  }
}
