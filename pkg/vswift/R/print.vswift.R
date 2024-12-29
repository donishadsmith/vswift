#' Print Parameter Information and/or Model Evaluation Metrics
#'
#' @name print
#'
#' @description Prints model configuration details and/or model evaluation metrics (classification accuracy, precision,
#' recall, and F1 scores) from a vswift object.
#'
#' @param x A vswift object.
#'
#' @param configs A logical value indicating whether to print model configuration information from the vswift
#' object. Default is \code{TRUE}.
#'
#' @param metrics A logical value indicating whether to print model evaluation metrics from the vswift object. If
#' \code{TRUE}, precision, recall, and F1 scores for each class will be displayed, along with their mean values
#' (if cross-validation was used). Default is \code{TRUE}.
#'
#' @param models A character string or a character vector specifying the classification algorithm(s) information to be
#' printed. If \code{NULL}, all model information will be printed. The following options are available:
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
#' @param ... No additional arguments are currently supported.
#'
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
#' @importFrom utils capture.output
#'
#' @author Donisha Smith
#'
#' @export

"print.vswift" <- function(x, configs = TRUE, metrics = TRUE, models = NULL, ...) {
  if (inherits(x, "vswift")) {
    # List for model names
    model_list <- list(
      "lda" = "Linear Discriminant Analysis", "qda" = "Quadratic Discriminant Analysis",
      "svm" = "Support Vector Machine", "nnet" = "Neural Network", "decisiontree" = "Decision Tree",
      "randomforest" = "Random Forest", "xgboost" = "Extreme Gradient Boosting",
      "multinom" = "Unregularized Multinomial Logistic Regression", "logistic" = "Unregularized Logistic Regression",
      "regularized_multinomial" = "Regularized Multinomial Logistic Regression",
      "regularized_logistic" = "Regularized Logistic Regression",
      "knn" = "K-Nearest Neighbors", "naivebayes" = "Naive Bayes"
    )

    # Get models
    if (is.null(models)) {
      models <- x$configs$models
    } else {
      models <- intersect(models, names(x$metrics))
    }

    if (length(models) == 0) stop("no valid models specified in `models` or `specified` models not found in `x$metrics`")

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
