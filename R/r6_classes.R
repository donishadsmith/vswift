#' Classification Results
#'
#' @name Vswift
#'
#' @description
#' An R6 class containing classification results produced by \code{\link{class_cv}}.
#' Provides methods for accessing metrics, trained models, data partitions,
#' and generating plots and curves.
#'
#' @examples
#' result <- class_cv(
#'   data = iris,
#'   target = "Species",
#'   models = c("svm", "lda"),
#'   train_params = list(split = 0.8, n_folds = 5, random_seed = 123)
#' )
#'
#' result$summary()
#' result$metrics("svm", "cv")
#' result$plot(metrics = "f1")
#'
#' @importFrom R6 R6Class
#' @export
Vswift <- R6Class("Vswift",
  public = list(
    #' @description Create a new vswift result object.
    #'
    #' @param configs List of configuration parameters.
    #'
    #' @param class_summary List with class-level info.
    #'
    #' @param metrics Named list of per-model metric dataframes.
    #'
    #' @param trained_models Named list of trained models.
    #'
    #' @param missing_data_summary Named list of missing data information.
    #'
    #' @param data_partitions List with indices and dataframes.
    #'
    #' @param imputation_models List of prep objects.
    initialize = function(configs, class_summary, metrics, trained_models,
                          missing_data_summary, data_partitions,
                          imputation_models) {
      private$.configs <- configs
      private$.class_summary <- class_summary
      private$.metrics <- metrics
      private$.trained_models <- trained_models
      private$.missing_data_summary <- missing_data_summary
      private$.data_partitions <- data_partitions
      private$.imputation_models <- imputation_models
    },
    #' @description Retrieve evaluation metrics.
    #'
    #' @param model Character. Model name. \code{NULL} returns all.
    #'
    #' @param type Character. "split" or "cv". \code{NULL} returns all for that
    #' model. Default is \code{NULL}.
    #'
    #' @return A data.frame or named list.
    metrics = function(model, type = NULL) {
      if (all(is.null(model), !is.null(type))) {
        stop("`model` cannot be `NULL` while type is not `NULL`")
      }

      if (is.null(model)) {
        return(private$.metrics)
      }

      obj <- .get_object(private$.metrics, model)
      if (is.null(type)) {
        return(obj)
      }

      return(.get_object(obj, type))
    },
    #' @description Retrieve configuration parameters.
    #'
    #' @param param Character. Config key. \code{NULL} returns all. Default
    #' is \code{NULL}.
    #'
    #' @param keys Character or list of characters. The sub-keys within param.
    #' \code{NULL} returns all keys of \code{param}. Default is \code{NULL}.
    #'
    #' @return The requested configuration value.
    configs = function(param = NULL, keys = NULL) {
      if (is.null(param)) {
        return(private$.configs)
      }

      obj <- .get_object(private$.configs, param)
      # NULL default needed for early return
      if (is.null(keys)) {
        return(obj)
      }


      for (key in c(keys)) obj <- .get_object(obj, key)

      return(obj)
    },
    #' @description Retrieve trained model objects.
    #'
    #' @param model Character. Model name. If \code{NULL}, returns all
    #' all models. Default is \code{NULL}.
    #'
    #' @param partition Character. "split", "final", or "fold1".."foldN".
    #' If \code{NULL}, returns all partitions. Default is \code{NULL}.
    #'
    #' @return A trained model object or named list.
    get_trained_model = function(model = NULL, partition = NULL) {
      if (is.null(private$.trained_models)) {
        return(NULL)
      }

      if (is.null(model)) {
        return(private$.trained_models)
      }

      obj <- .get_object(private$.trained_models, model)

      if (is.null(partition)) {
        return(obj)
      }

      return(.get_object(obj, partition))
    },
    #' @description Retrieve the imputation objects.
    #'
    #' @param model Character. Model name. If \code{NULL}, returns all
    #' all models. Default is \code{NULL}.
    #'
    #' @param partition Character. "split", "final", or "fold1".."foldN".
    #' If \code{NULL}, returns all partitions. Default is \code{NULL}.
    #'
    #' @return An imputation model object or named list.
    get_imputation_model = function(model = NULL, partition = NULL) {
      if (is.null(private$.imputation_models)) {
        return(NULL)
      }

      if (is.null(model)) {
        return(private$.imputation_models)
      }

      obj <- .get_object(private$.imputation_models, model)

      if (is.null(partition)) {
        return(obj)
      }

      return(.get_object(obj, partition))
    },
    #' @description Retrieve missing data summary.
    #'
    #' @param what Character. The specific missing data information.
    #' \code{NULL} returns all. Default is \code{NULL}.
    #'
    #' @return The requested missing data information.
    get_missing_data_summary = function(what = NULL) {
      if (is.null(what)) {
        return(private$.missing_data_summary)
      }

      return(.get_object(private$.missing_data_summary, what))
    },
    #' @description Retrieve data partition information.
    #'
    #' @param what Character. "indices", "proportions", or "dataframes".
    #' Default is \code{NULL}.
    #'
    #' @param partition Character. "split" or "fold1".."foldN". If \code{NULL},
    #' returns all partitions. Default is \code{NULL}.
    #'
    #' @param set Character. "train" or "test". \code{NULL}, returns the
    #' training and test set. Default is \code{NULL}.
    #'
    #' @return Requested partition data.
    get_partition = function(what = NULL, partition = NULL, set = NULL) {
      if (is.null(what)) {
        return(private$.data_partitions)
      }

      obj <- .get_object(private$.data_partitions, what)

      if (is.null(partition)) {
        return(obj)
      }

      obj <- .get_object(obj, partition)

      if (is.null(set)) {
        return((obj))
      }

      return(.get_object(obj, set))
    },
    #' @description Retrieve class summary information.
    #'
    #' @param what Character. "classes", "keys", "proportions", or "indices".
    #'
    #' @return The requested class summary component.
    class_info = function(what) {
      return(.get_object(private$.class_summary, what))
    },
    #' @description Prints model configuration details and/or model evaluation
    #' metrics (classification accuracy, precision, recall, and F1 scores).
    #'
    #' @param configs A logical value indicating whether to print model
    #' configuration information from the vswift class. Default is \code{TRUE}.
    #'
    #' @param metrics A logical value indicating whether to print model
    #' evaluation metrics from the vswift class If \code{TRUE}, precision,
    #' recall, and F1 scores for each class will be displayed, along with their
    #' mean values (if cross-validation was used). Default is \code{TRUE}.
    #'
    #' @param models A character string or a character vector specifying the
    #' classification algorithm(s) information to be printed. If \code{NULL},
    #' all model information will be printed. The following options ar
    #' available:
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
    #'  \item \code{"regularized_multinomial"}: Regularized Multinomial Logistic
    #'  Regression
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
    #' results <- class_cv(
    #'   data = iris,
    #'   target = "Species",
    #'   models = "lda",
    #'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 123)
    #' )
    #'
    #' # Print parameter information and performance metrics
    #' results$print()
    #'
    #' @importFrom utils capture.output
    print = function(configs = TRUE, metrics = TRUE, models = NULL) {
      # Calculate string length of classes
      str_list <- .dashed_lines(self$classes, TRUE)
      for (model in private$.resolve_models(models)) {
        cat(paste("Model:", .MODEL_LIST[[model]]), "\n\n")
        if (configs) .print_configs(self, model)

        kwargs <- list(
          x = self, model = model, max_str_len = str_list$max
        )

        if (metrics) {
          if (self$has_split) do.call(.print_metrics_split, kwargs)
          if (self$has_cv) do.call(.print_metrics_cv, kwargs)
        }

        .dashed_lines(self$classes)
      }

      invisible(self)
    },
    #' @description Plots classification metrics (accuracy, precision, recall,
    #' and f1 for each class).
    #'
    #' @param metrics A character vector indicating which metrics to plot.
    #' Supported options are "accuracy", "recall", "precision", "f1". Default is
    #' \code{c("accuracy", "precision", "recall", "f1")}.
    #'
    #' @param models A character string or a character vector specifying the
    #' classification algorithm(s) evaluation metrics to plot. If \code{NULL},
    #' all models will be plotted. The following options are available:
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
    #'  \item \code{"regularized_multinomial"}: Regularized Multinomial Logistic
    #'  Regression
    #'  \item \code{"xgboost"}: Extreme Gradient Boosting
    #'  }
    #'  Default = \code{NULL}.
    #'
    #' @param split A logical value indicating whether to plot metrics for the
    #' train-test split results. Default is \code{TRUE}.
    #'
    #' @param cv A logical value indicating whether to plot metrics for
    #' cross-validation results. Default is \code{TRUE}.
    #'
    #' @param class_names A vector of the specific classes to plot. If
    #' \code{NULL}, plots are generated for all classes. Default is \code{NULL}.
    #'
    #' @param path A character string specifying the directory (with a trailing
    #' slash) to save the plots.
    #' Default is \code{NULL}.
    #'
    #' @param ... Additional arguments passed to the \code{png} function.
    #'
    #' @examples
    #' # Load an example dataset
    #' data(iris)
    #'
    #' # Perform a train-test split with an 80% training set and stratified
    #' # sampling using QDA
    #' results <- class_cv(
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
    #' results$plot(class_names = "setosa", metrics = "f1")
    #'
    #' @importFrom grDevices dev.off dev.new graphics.off png
    #' @importFrom  graphics axis abline legend
    plot = function(metrics = c("accuracy", "precision", "recall", "f1"),
                    models = NULL, split = TRUE, cv = TRUE, class_names = NULL,
                    path = NULL, ...) {
      valid_metrics <- c("accuracy", "precision", "recall", "f1")
      metrics <- intersect(unlist(lapply(metrics, tolower)), valid_metrics)
      if (length(metrics) == 0) {
        met_str <- paste(valid_metrics, collapse = ", ")
        stop(
          sprintf("no metrics specified, available metrics: %s", met_str)
        )
      }
      if (!is.null(class_names)) {
        class_names <- intersect(class_names, self$classes)
        if (length(class_names) == 0) {
          class_str <- paste(self$classes, collapse = ", ")
          stop(
            sprintf("no classes specified, available classes: %s", class_str)
          )
        }
      }

      for (model in private$.resolve_models(models)) {
        .plot(
          x = self, metrics = metrics, model = model,
          plot_title = .MODEL_LIST[[model]], split = split, cv = cv,
          class_names = class_names, path = path, ...
        )
      }

      invisible(self)
    },
    #' @description Print a compact summary of results.
    summary = function() {
      cat("Classification Results\n")
      cat("-----------------------------\n")
      cat("  Models:  ", paste(self$available_models, collapse = ", "), "\n")
      cat("  Classes: ", paste(self$classes, collapse = ", "), "\n")
      if (self$has_split) {
        train_split <- self$configs("train_params", "split")
        test_split <- 1 - train_split
        cat("  Split:   ", sprintf("%s (Training), %s (Test)", train_split, test_split), "\n")
      }
      if (self$has_cv) cat("  Folds:   ", self$n_folds, "\n")

      if (self$has_split) cat("\n  Mean Classification Accuracy (Train-Test Split):\n")
      for (model in self$available_models) {
        split_df <- self$metrics(model, "split")
        if (is.data.frame(split_df)) {
          train_acc <- split_df[split_df$Set == "Training", "Classification Accuracy"]
          test_acc <- split_df[split_df$Set == "Test", "Classification Accuracy"]
          cat(
            sprintf(
              "    %-30s %.3f (Training),  %.3f (Test)\n",
              .MODEL_LIST[[model]], train_acc, test_acc
            )
          )
        }
      }

      cat("\n  Mean Classification Accuracy (CV):\n")
      for (model in self$available_models) {
        cv_df <- self$metrics(model, "cv")
        if (is.data.frame(cv_df)) {
          acc <- cv_df[cv_df$Fold == "Mean CV:", "Classification Accuracy"]
          cat(sprintf("    %-30s %.3f\n", .MODEL_LIST[[model]], acc))
        }
      }
    },
    #' @description Produces ROC curves and computes the area under the curve
    #' (AUC) and Youden's Index. Only works for binary classification tasks.
    #'
    #' @param data A data frame. If \code{NULL}, then the preprocessed data must
    #' be saved using \code{save = list("data" = TRUE)} in \code{class_cv}.
    #' Default = \code{NULL}.
    #'
    #' @param models A character string or a character vector specifying the
    #' classification algorithm(s) to plot curves for. If \code{NULL}, all
    #' models will be plotted. The following options are available:
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
    #'  \item \code{"regularized_multinomial"}: Regularized Multinomial Logistic
    #'  Regression
    #'  \item \code{"xgboost"}: Extreme Gradient Boosting
    #'  }
    #'  Default = \code{NULL}.
    #'
    #' @param split A logical value indicating whether to plot curves for the
    #' train-test split results. Default is \code{TRUE}.
    #'
    #' @param cv A logical value indicating whether to plot curves for
    #' cross-validation results. Default is \code{TRUE}.
    #'
    #' @param thresholds A numerical vector specifying the thresholds to use
    #' when producing the curves. If left as \code{NULL} the unique probability
    #' values produced by the training model will be used as thresholds.
    #' Default is \code{NULL}.
    #'
    #' @param return_output A logical value indicating whether to return the
    #' output list. Default is \code{TRUE}.
    #'
    #' @param path A character string specifying the directory (with a trailing
    #' slash) to save the plots. Default is \code{NULL}.
    #'
    #' @param ... Additional arguments passed to the \code{png} function.
    #'
    #' @return A \code{\link{CurveResult}} object containing thresholds, target
    #' labels, false positive rates (FPR), true positive rates (TPR), area under
    #' the curve (AUC), and Youden's Index for all training and validation sets
    #' for each model.
    #'
    #' @examples
    #' # Load an example dataset
    #' data <- iris
    #'
    #' # Make Binary
    #' data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
    #'
    #' # Perform a train-test split with an 80% training set and stratified
    #' # sampling using QDA
    #' results <- class_cv(
    #'   data = data,
    #'   target = "Species",
    #'   models = "qda",
    #'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 123),
    #'   save = list(data = TRUE, models = TRUE)
    #' )
    #'
    #' # Get ROC curve
    #' results$roc_curve(return_output = FALSE)
    #'
    #' @importFrom grDevices rainbow
    #' @importFrom graphics lines
    roc_curve = function(data = NULL,
                         models = NULL,
                         split = TRUE,
                         cv = TRUE,
                         thresholds = NULL,
                         return_output = TRUE,
                         path = NULL, ...) {
      models <- private$.resolve_models(models)
      results <- .curve_entry(
        self, data, models, split, cv, thresholds, return_output,
        "roc", path, ...
      )

      return(CurveResult$new(results, "roc"))
    },
    #' @description Produces PR curves and computes the area under the curve
    #' (AUC) and the threshold with the maximum F1 score. Only works for binary
    #' classification tasks.
    #'
    #' @param data A data frame. If \code{NULL}, then the preprocessed data
    #' must be saved using  \code{save = list("data" = TRUE)} in \code{class_cv}.
    #' Default = \code{NULL}.
    #'
    #' @param models A character string or a character vector specifying the
    #' classification algorithm(s) to plot curves for. If \code{NULL}, all
    #' models will be plotted. The following options are available:
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
    #'  \item \code{"regularized_multinomial"}: Regularized Multinomial Logistic
    #'  Regression
    #'  \item \code{"xgboost"}: Extreme Gradient Boosting
    #'  }
    #'  Default = \code{NULL}.
    #'
    #' @param split A logical value indicating whether to plot curves for the
    #' train-test split results. Default is \code{TRUE}.
    #'
    #' @param cv A logical value indicating whether to plot curves for
    #' cross-validation results. Default is \code{TRUE}.
    #'
    #' @param thresholds A numerical vector specifying the thresholds to use
    #' when producing the curves. If left as \code{NULL} the unique probability
    #' values produced by the training model will be used as thresholds.
    #' Default is \code{NULL}.
    #'
    #' @param return_output A logical value indicating whether to return the
    #' output list. Default is \code{TRUE}.
    #'
    #' @param path A character string specifying the directory (with a trailing
    #' slash) to save the plots.
    #' Default is \code{NULL}.
    #'
    #' @param ... Additional arguments passed to the \code{png} function.
    #'
    #' @return A \code{\link{CurveResult}} object containing thresholds, target
    #' labels, precision, recall, area under the curve (AUC), and maximum F1
    #' score and its associated optimal threshold for all training and validation
    #' sets for each model.
    #'
    #' @examples
    #' # Load an example dataset
    #' data <- iris
    #'
    #' # Make Binary
    #' data$Species <- ifelse(data$Species == "setosa", "setosa", "not setosa")
    #'
    #' # Perform a train-test split with an 80% training set and stratified
    #' # sampling using QDA
    #' results <- class_cv(
    #'   data = data,
    #'   target = "Species",
    #'   models = "qda",
    #'   train_params = list(split = 0.8, stratified = TRUE, random_seed = 123),
    #'   save = list(data = TRUE, models = TRUE)
    #' )
    #'
    #' # Get PR curve
    #' results$pr_curve(return_output = FALSE)
    pr_curve = function(data = NULL,
                        models = NULL,
                        split = TRUE,
                        cv = TRUE,
                        thresholds = NULL,
                        return_output = TRUE,
                        path = NULL,
                        ...) {
      models <- private$.resolve_models(models)
      results <- .curve_entry(
        self, data, models, split, cv, thresholds, return_output, "pr",
        path, ...
      )

      return(CurveResult$new(results, "pr"))
    }
  ),
  private = list(
    .configs = NULL,
    .class_summary = NULL,
    .metrics = NULL,
    .trained_models = NULL,
    .missing_data_summary = NULL,
    .data_partitions = NULL,
    .imputation_models = NULL,
    .resolve_models = function(target_models) {
      # Get models
      if (is.null(target_models)) {
        models <- self$available_models
      } else {
        # Make lowercase
        models <- sapply(target_models, tolower)
        models <- intersect(models, self$available_models)

        if (length(models) == 0) stop("no valid models specified in `models`")

        if (length(invalid_models <- setdiff(models, target_models)) > 0) {
          warning(
            sprintf(
              "invalid model in models or information for specified model
              not present in vswift x: %s",
              paste(unlist(invalid_models), collapse = ", ")
            )
          )
        }
      }

      return(models)
    }
  ),
  active = list(
    #' @field classes Character vector of target classes.
    classes = function() private$.class_summary$classes,
    #' @field n_models Number of models in this result.
    n_models = function() length(private$.metrics),
    #' @field available_models Character vector of model names.
    available_models = function() self$configs("models"),
    #' @field has_split TRUE if train-test split was performed.
    has_split = function() !is.null(private$.configs$train_params$split),
    #' @field has_cv TRUE if cross-validation was performed.
    has_cv = function() !is.null(private$.configs$train_params$n_folds),
    #' @field n_folds Number of CV folds. NULL if no CV.
    n_folds = function() private$.configs$train_params$n_folds
  )
)


#' Curve Results
#'
#' @name CurveResult
#'
#' @description
#' An R6 class containing ROC or Precision-Recall curve results produced by
#' the \code{roc_curve} or \code{pr_curve} methods of a \code{\link{Vswift}}
#' object. Provides methods for accessing probabilities, AUC, optimal
#' thresholds, and comparing models.
#'
#' @export
CurveResult <- R6::R6Class(
  "CurveResult",
  public = list(
    #' @description Create a new CurveResult object.
    #'
    #' @param results Named list of curve results keyed by model name.
    #'
    #' @param curve_type Character. Either "roc" or "pr".
    initialize = function(results, curve_type) {
      private$.results <- results
      private$.type <- curve_type
    },

    #' @description Retrieve curve results for a specific model.
    #'
    #' @param model Character. Model name. \code{NULL} returns all models.
    #' Default is \code{NULL}.
    #'
    #' @return A named list of curve results, or all results if \code{model}
    #' is \code{NULL}.
    get_model = function(model = NULL) {
      if (is.null(model)) {
        return(private$.results)
      }

      return(.get_object(private$.results, model))
    },

    #' @description Retrieve predicted probabilities for a model partition.
    #'
    #' @param model Character. Model name.
    #'
    #' @param partition Character. "split" or "fold1".."foldN".
    #'
    #' @param set Character. "train" or "test".
    #'
    #' @return A numeric vector of predicted probabilities.
    get_probs = function(model, partition, set) {
      obj <- .get_object(private$.results, model)
      obj <- .get_object(obj, partition)

      return(.get_object(obj, set)$probs)
    },

    #' @description Retrieve the area under the curve (AUC).
    #'
    #' @param model Character. Model name. If \code{NULL}, returns AUC for
    #' all models as a named vector. Default is \code{NULL}.
    #'
    #' @param partition Character. "split" or "fold1".."foldN". Default is
    #' \code{"split"}.
    #'
    #' @param set Character. "train" or "test". Default is \code{"test"}.
    #'
    #' @return A numeric value or named numeric vector of AUC values.
    get_auc = function(model = NULL, partition = "split", set = "test") {
      if (!is.null(model)) {
        obj <- .get_object(private$.results, model)
        obj <- .get_object(obj, partition)

        return(.get_object(obj, set)$auc)
      }

      auc <- sapply(names(private$.results), function(x) {
        if (partition == "split") {
          private$.results[[x]]$split[[set]]$auc
        } else {
          private$.results[[x]]$cv[[partition]]$auc
        }
      })

      return(auc)
    },

    #' @description Retrieve the maximum F1 score. Only available for
    #' Precision-Recall curves.
    #'
    #' @param model Character. Model name. If \code{NULL}, returns max F1 for
    #' all models as a named vector. Default is \code{NULL}.
    #'
    #' @param partition Character. "split" or "fold1".."foldN". Default is
    #' \code{"split"}.
    #'
    #' @param set Character. "train" or "test". Default is \code{"test"}.
    #'
    #' @return A numeric value, named numeric vector, or \code{NULL} if the
    #' curve type is not "pr".
    get_max_f1 = function(model = NULL, partition = "split", set = "test") {
      if (private$.type != "pr") {
        return(NULL)
      }

      if (!is.null(model)) {
        obj <- .get_object(private$.results, model)
        obj <- .get_object(obj, partition)

        return(.get_object(obj, set)$max_f1)
      }

      max_f1 <- sapply(names(private$.results), function(x) {
        if (partition == "split") {
          private$.results[[x]]$split[[set]]$max_f1
        } else {
          private$.results[[x]]$cv[[partition]]$max_f1
        }
      })

      return(max_f1)
    },

    #' @description Retrieve the optimal threshold. For ROC curves, this is
    #' Youden's Index. For PR curves, this is the threshold that maximizes the
    #' F1 score.
    #'
    #' @param model Character. Model name. If \code{NULL}, returns optimal
    #' thresholds for all models as a named vector. Default is \code{NULL}.
    #'
    #' @param partition Character. "split" or "fold1".."foldN". Default is
    #' \code{"split"}.
    #'
    #' @param set Character. "train" or "test". Default is \code{"test"}.
    #'
    #' @return A numeric value or named numeric vector of optimal thresholds.
    get_optimal_threshold = function(model = NULL, partition = "split",
                                     set = "test") {
      metric <- ifelse(
        private$.type == "pr", "optimal_threshold", "youdens_index"
      )

      if (!is.null(model)) {
        obj <- .get_object(private$.results, model)
        obj <- .get_object(obj, partition)

        return(.get_object(obj, set)[[metric]])
      }

      optimal_threshold <- sapply(names(private$.results), function(x) {
        if (partition == "split") {
          private$.results[[x]]$split[[set]][[metric]]
        } else {
          private$.results[[x]]$cv[[partition]][[metric]]
        }
      })

      return(optimal_threshold)
    },

    #' @description Retrieve curve metrics (FPR/TPR for ROC, or
    #' precision/recall for PR curves).
    #'
    #' @param model Character. Model name.
    #'
    #' @param partition Character. "split" or "fold1".."foldN". Default is
    #' \code{"split"}.
    #'
    #' @param set Character. "train" or "test". Default is \code{"test"}.
    #'
    #' @return A named list containing the curve metrics.
    get_metrics = function(model, partition = "split", set = "test") {
      obj <- .get_object(private$.results, model)
      obj <- .get_object(obj, partition)

      return(.get_object(obj, set)$metrics)
    },

    #' @description Compare AUC across all models for a given partition.
    #'
    #' @param partition Character. "split" or "fold1".."foldN".
    #'
    #' @param set Character. "train" or "test".
    #'
    #' @return A data.frame with columns \code{model} and \code{auc}.
    compare = function(partition, set) {
      models <- names(private$.results)
      data.frame(
        model = models,
        auc = sapply(models, function(m) self$get_auc(m, partition, set)),
        row.names = NULL
      )
    }
  ),
  private = list(
    .results = NULL,
    .type = NULL
  )
)

.get_object <- function(x, name) {
  if (!name %in% names(x)) {
    valid_names <- names(x)
    valid_name_str <- paste(valid_names, collapse = ", ")
    stop(
      sprintf(
        "'%s' is not a valid name, valid names are: %s",
        name, valid_name_str
      )
    )
  }

  return(x[[name]])
}
