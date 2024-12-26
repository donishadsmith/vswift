#' Perform Train-Test Splitting and/or Cross-Validation on Classification Data
#'
#' @name classCV
#'
#' @description Performs train-test splitting and/or cross-validation on classification data using various
#'              classification algorithms.
#'
#' @param data A data frame. Default = \code{NULL}
#'
#' @param formula A formula specifying the model to use. This argument cannot be used when \code{target}
#'                (and optionally \code{predictors}) is specified. Default is \code{NULL}.
#'
#' @param target The name or numerical index of the target (response) variable in \code{data}. This argument cannot be
#'               used when \code{formula} is specified. Default is \code{NULL}.
#'
#' @param predictors A vector of variable names or numerical indices indicating the predictors in \code{data},
#'                   used in conjunction with \code{target}. Default is \code{NULL}.
#'
#' @param models A character string or a character vector specifying the classification algorithm(s) to use.
#'               The following options are available:
#'               \itemize{
#'                 \item \code{"lda"}: Linear Discriminant Analysis
#'                 \item \code{"qda"}: Quadratic Discriminant Analysis
#'                 \item \code{"logistic"}: Logistic Regression (unregularized)
#'                 \item \code{"svm"}: Support Vector Machine
#'                 \item \code{"naivebayes"}: Naive Bayes
#'                 \item \code{"nnet"}: Neural Network
#'                 \item \code{"knn"}: K-Nearest Neighbors
#'                 \item \code{"decisiontree"}: Decision Tree
#'                 \item \code{"randomforest"}: Random Forest
#'                 \item \code{"multinom"}: Multinomial Logistic Regression (unregularized)
#'                 \item \code{"xgboost"}: Extreme Gradient Boosting
#'               }
#'
#'               \strong{Notes:}
#'               \itemize{
#'                 \item \code{"knn"}: The \code{ks} parameter should be set to specify the desired value of \emph{k},
#'                                     ensuring that the same value is used in all folds. If \code{ks} is not provided,
#'                                     the optimal \emph{k} is automatically selected using the \pkg{kknn} package.
#'
#'                 \item \code{"nnet"}: An additional argument \code{size} must be specified.
#'
#'                 \item \code{"xgboost"}: The following \code{objective} functions are supported:
#'                                       \code{"reg:logistic"}, \code{"binary:logistic"}, \code{"binary:logitraw"},
#'                                       \code{"binary:hinge"}, and \code{"multi:softprob"}.
#'               }
#'
#' @param model_params A list that can include the following elements:
#'                     \itemize{
#'                       \item \code{"map_args"}: A list of named sub-lists used when more than one model is specified
#'                                                in \code{models}. Each sub-list corresponds to a particular model in
#'                                                the \code{models} parameter and contains the arguments that will be
#'                                                passed to that model. Default is \code{NULL}. Refer to the
#'                                                "Additional Model Parameters" section for acceptable arguments.
#'
#'                       \item \code{"logistic_threshold"}: A numeric value between 0 and 1, serving as the decision
#'                                                          boundary for logistic regression. Observations are assigned
#'                                                          to the class coded as "1" if
#'                                                          \code{P(Class = 1 | Features) >= logistic_threshold};
#'                                                          otherwise, they are assigned to the class coded as "0".
#'                                                          This threshold is used when \code{"logistic"} is included
#'                                                          in \code{models}, or when \code{"xgboost"} is included in
#'                                                          \code{models} with one of these objective functions:
#'                                                          \code{"reg:logistic"}, \code{"binary:logistic"}, or
#'                                                          \code{"binary:logitraw"}. Default is \code{0.5}.
#'
#'                       \item \code{"final_model"}: A logical value indicating whether to use all complete observations
#'                                                   in the input data for model training. Default is \code{FALSE}.
#'                     }
#'
#' @param train_params A list that can contain the following parameters:
#'                     \itemize{
#'                       \item \code{split}: A numeric value between 0 and 1 indicating the proportion of data to use
#'                                           for training. The remaining observations are allocated to the test set. If
#'                                           not specified or set to \code{NULL}, no train-test splitting is performed.
#'                                           Note that this split is separate from cross-validation. Default is
#'                                           \code{NULL}.
#'
#'                       \item \code{n_folds}: An integer greater than 2 specifying the number of folds for
#'                                             cross-validation. If \code{NULL}, no cross-validation is performed.
#'                                             Default is \code{NULL}.
#'
#'                       \item \code{stratified}: A logical value indicating whether stratified sampling should be used
#'                                                during splitting. Default is \code{FALSE}.
#'
#'                       \item \code{random_seed}: A numeric value for the random seed to ensure reproducibility of
#'                                                 random splitting and any model training that relies on random starts.
#'                                                 Default is \code{NULL}.
#'
#'                       \item \code{standardize}: A logical or a numeric/character vector. If \code{TRUE}, all numeric
#'                                                 columns (except the target) are standardized by computing the mean
#'                                                 and standard deviation from the training subset and applying them to
#'                                                 both the training and test/validation sets. This prevents data
#'                                                 leakage. A vector of column indices or names can also be provided to
#'                                                 only standardize specific columns.
#'
#'                       \item \code{remove_obs}: A logical value indicating whether to remove observations in the
#'                                                test/validation set that contain levels of categorical predictors not
#'                                                seen in the training data. Some algorithms may produce errors when
#'                                                encountering such levels in the validation data during prediction.
#'                                                Default is \code{FALSE}.
#'                     }
#'
#' @param impute_params A list defining how to handle missing values among predictors/features.
#'                      During imputation, the target variable is excluded from both training and
#'                      test/validation sets. Prior to imputation, unlabeled data (observations with
#'                      missing targets) are removed, and any specified train-test split or cross-validation
#'                      folds are created. A separate imputation model is then generated for each training
#'                      subset (one for the train-test split and one per fold). Each imputation model is
#'                      applied to both its corresponding training and test/validation subsets to minimize
#'                      data leakage.
#'
#'                      Note that numerical columns are automatically standardized (regardless of
#'                      \code{train_params$standardize}) before imputation occurs. The \pkg{recipes} package is used for
#'                      imputation. The following parameters are available:
#'
#'                      \itemize{
#'                        \item \code{method}: A character specifying the imputation method. Options include:
#'                                             \itemize{
#'                                               \item \code{"bag_impute"}: Bagged Trees Imputation
#'                                               \item \code{"knn_impute"}: K-Nearest Neighbors Imputation
#'                                             }
#'                                             Default is \code{NULL}.
#'
#'                       \item \code{args}: A list of additional arguments for the chosen imputation method.
#'                                          \itemize{
#'                                            \item \code{"bag_impute"}: \code{neighbors}
#'                                            \item \code{"knn_impute"}: \code{trees}, \code{seed_val}
#'                                          }
#'                                          For more details about these arguments, consult the \pkg{recipes}
#'                                          documentation. Default is \code{NULL}.
#'   }
#'
#' @param save A list that may include the following:
#'             \itemize{
#'               \item \code{models}: A logical value indicating whether to save the trained models
#'                                    (including imputation models) used for train-test splits or cross-validation.
#'                                    Default is \code{FALSE}.
#'
#'               \item \code{data}: A logical value indicating whether to save all training and test/validation sets
#'                                  used during train-test splitting and/or cross-validation. Default is \code{FALSE}.
#'             }
#'
#' @param parallel_configs A list that may include the following:
#'                         \itemize{
#'                           \item \code{n_cores}: A numeric value specifying the number of cores for parallel
#'                                                 processing. Default is \code{NULL}.
#'
#'                           \item \code{future.seed}: A numeric value indicating the seed to use with \pkg{future} for
#'                                                     parallel processing.
#'                         }
#'
#' @param ... Additional arguments for the chosen classification algorithm. These arguments serve as an alternative
#'            to specifying model-specific parameters in \code{model_params$map_args} when only a single
#'            model is specified in \code{models}. If multiple models are specified, then \code{map_args} must be used.
#'            Refer to each algorithm's documentation for details on additional arguments.
#'
#' @section Additional Model Parameters:
#'          Each element in \code{models} accepts arguments specific to its underlying classification algorithm.
#'          Refer to the original package documentation for more information about these arguments.
#'          Further details on the external package functions used for each model are provided in the
#'          "Package Dependencies" section. The available arguments for each \code{models} value are:
#'           \itemize{
#'             \item \code{"lda"}: \code{prior}, \code{method}, \code{nu}, \code{tol}
#'             \item \code{"qda"}: \code{prior}, \code{method}, \code{nu}
#'             \item \code{"logistic"}: \code{weights}, \code{singular.ok}, \code{maxit}
#'             \item \code{"svm"}: \code{kernel}, \code{degree}, \code{gamma}, \code{cost}, \code{nu},
#'                                 \code{class.weights}, \code{shrinking}, \code{epsilon}, \code{tolerance},
#'                                 \code{cachesize}
#'             \item \code{"naivebayes"}: \code{prior}, \code{laplace}, \code{usekernel}, \code{bw}, \code{kernel},
#'                                        \code{adjust}, \code{weights}, \code{give.Rkern}, \code{subdensity},
#'                                        \code{from}, \code{to}, \code{cut}
#'             \item \code{"nnet"}: \code{size}, \code{rang}, \code{decay}, \code{maxit}, \code{softmax},
#'                                  \code{entropy}, \code{abstol}, \code{reltol}, \code{Hess}, \code{skip}
#'             \item \code{"knn"}: \code{kmax}, \code{ks}, \code{distance}, \code{kernel}
#'             \item \code{"decisiontree"}: \code{weights}, \code{method},\code{parms}, \code{control}, \code{cost}
#'             \item \code{"randomforest"}: \code{weights}, \code{ntree}, \code{mtry}, \code{nodesize},
#'                                          \code{importance}, \code{localImp}, \code{nPerm}, \code{proximity},
#'                                          \code{keep.forest}, \code{norm.votes}
#'             \item \code{"multinom"}: \code{weights}, \code{Hess}
#'             \item \code{"xgboost"}: \code{params}, \code{nrounds}, \code{print_every_n}, \code{feval},
#'                                     \code{verbose}, \code{early_stopping_rounds}, \code{obj}, \code{save_period},
#'                                     \code{save_name}
#'           }
#'
#' @section Package Dependencies:
#'          Each option of \code{models} uses the following function from the specified packages:
#'   \itemize{
#'     \item \code{"lda"}: \code{lda} from \pkg{MASS} package
#'     \item \code{"qda"}: \code{qda} from \pkg{MASS} package
#'     \item \code{"logistic"}: \code{glm} from \pkg{base} package with \code{family = "binomial"}
#'     \item \code{"svm"}: \code{svm()} from \pkg{e1071} package
#'     \item \code{"naivebayes"}: \code{naive_bayes} from \pkg{naivebayes} package
#'     \item \code{"nnet"}: \code{nnet} from \pkg{nnet} package
#'     \item \code{"knn"}: \code{train.kknn} from \pkg{kknn} package
#'     \item \code{"decisiontree"}: \code{rpart} from \pkg{rpart} package
#'     \item \code{"randomforest"}: \code{randomForest} from \pkg{randomForest} package
#'     \item \code{"multinom"}: \code{multinom} from \pkg{nnet} package
#'     \item \code{"xgboost"}: \code{xgb.train} from \pkg{xgboost} package
#'   }
#'
#' @return A list (vswift object) containing:
#'   \itemize{
#'     \item Any train-test split or cross-validation results (if specified).
#'     \item Performance metrics.
#'     \item Class distribution details for the training set, test set, and folds (if applicable).
#'     \item Saved models (if requested).
#'     \item Saved datasets (if requested).
#'     \item A final model (if requested).
#'   }
#'
#' @seealso \code{\link{print.vswift}}, \code{\link{plot.vswift}}
#'
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#' result <- classCV(
#'   data = iris,
#'   target = "Species",
#'   models = "lda",
#'   train_params = list(split = 0.8)
#' )
#'
#' # Print parameters and metrics
#' result
#'
#' # Perform 5-fold cross-validation using Extreme Gradient Boosting
#' # w/ additional parameters: params & nrounds
#' result <- classCV(
#'   data = iris,
#'   formula = Species ~ .,
#'   models = "xgboost",
#'   train_params = list(n_folds = 5, random_seed = 50),
#'   params = list(
#'     objective = "multi:softprob",
#'     num_class = 3,
#'     eta = 0.3,
#'     max_depth = 6
#'   ),
#'   nrounds = 10
#' )
#'
#' # Print parameters and metrics
#' result
#'
#'
#' # Perform 5-fold cross-validation a train-test split w/multiple models
#' args <- list("knn" = list(ks = 5), "nnet" = list(size = 20))
#' result <- classCV(
#'   data = iris,
#'   target = 5,
#'   predictors = c(1:3),
#'   models = c("decisiontree", "knn", "nnet", "svm"),
#'   model_params = list(map_args = args),
#'   train_params = list(
#'     n_folds = 5,
#'     stratified = TRUE,
#'     random_seed = 50
#'   )
#' )
#'
#' # Print parameters and metrics
#' result
#'
#' @author Donisha Smith
#'
#' @importFrom stats as.formula complete.cases glm predict sd
#'
#' @export
classCV <- function(data,
                    formula = NULL,
                    target = NULL,
                    predictors = NULL,
                    models,
                    model_params = list("map_args" = NULL, "logistic_threshold" = 0.5, "final_model" = FALSE),
                    train_params = list(
                      "split" = NULL, "n_folds" = NULL, "stratified" = FALSE,
                      "random_seed" = NULL, "standardize" = FALSE, "remove_obs" = FALSE
                    ),
                    impute_params = list("method" = NULL, "args" = NULL),
                    save = list("models" = FALSE, "data" = FALSE),
                    parallel_configs = list("n_cores" = NULL, "future.seed" = NULL),
                    ...) {
  # Ensure model type is lowercase
  if (!is.null(models)) models <- tolower(models)

  # Ensure model types are unique
  models <- unique(models)

  # Append arguments; append missing so that default arguments appear in the output list and in order
  model_params <- .append_keys("model_params", model_params, models, ...)
  train_params <- .append_keys("train_params", train_params)
  impute_params <- .append_keys("impute_params", impute_params)
  save <- .append_keys("save", save)
  parallel_configs <- .append_keys("parallel_configs", parallel_configs)

  # Checking if inputs are valid
  .error_handling(
    data = data, formula = formula, target = target, predictors = predictors, models = models,
    model_params = model_params, train_params = train_params, impute_params = impute_params,
    save = save, parallel_configs = parallel_configs, call = "classCV"
  )

  # Get character form of target and predictor variables
  vars <- .get_var_names(formula, target, predictors, data)

  # Get information on unlabeled data and labeled data with missing features
  missing_info <- .missing_summary(data, vars$target)

  # Clean data; Unlabeled data dropped and labeled missing data dropped if imputation is not requested
  clean_outputs <- .clean_data(data, missing_info, !is.null(impute_params$method))
  preprocessed_data <- clean_outputs$cleaned_data
  perform_imputation <- clean_outputs$perform_imputation

  # Ensure target is factored and get all levels of character columns obtained if svm in models
  factored <- .convert_to_factor(preprocessed_data, vars$target, models, train_params)
  preprocessed_data <- factored$data
  col_levels <- factored$col_levels

  # Delete data
  rm(data, factored)
  gc()

  # Store information
  final_output <- .store_parameters(
    formula, missing_info, preprocessed_data, vars, models, model_params, train_params,
    impute_params, save, parallel_configs
  )

  # Create class dictionary
  if (any(models %in% c("logistic", "xgboost"))) {
    final_output$class_summary$keys <- .create_dictionary(preprocessed_data[, vars$target])
  }

  # Sampling data
  if (!is.null(train_params$split) | !is.null(train_params$n_folds)) {
    # Initialize list to store sample indices
    final_output$data_partitions <- list()
    final_output <- .sampling(preprocessed_data, train_params, vars$target, final_output)
    # Create the empty dataframes for metrics
    final_output$metrics <- .expand_dataframe(train_params, models, final_output$class_summary$classes)
    # Partition data to ensure no issues with floating point precision or stochastic imputations
    df_list <- .partition(preprocessed_data, final_output$data_partitions$indices)
  }

  # Generate vector for iteration
  iters <- .gen_iterations(train_params, model_params)

  # Impute train and test data
  if (!is.null(impute_params$method) && perform_imputation) {
    for (i in iters) {
      if (i == "split" && exists("df_list")) {
        prep_out <- .prep_data(
          train = df_list$split$train, test = df_list$split$test, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        df_list$split <- prep_out[!names(prep_out) == "prep"]
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$split <- prep_out$prep
      } else if (startsWith(i, "fold") && exists("df_list")) {
        prep_out <- .prep_data(
          train = df_list$cv[[i]]$train, test = df_list$cv[[i]]$test, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        df_list$cv[[i]] <- prep_out[!names(prep_out) == "prep"]
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$cv[[i]] <- prep_out$prep
      } else {
        prep_out <- .prep_data(
          preprocessed_data = preprocessed_data, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        preprocessed_data <- prep_out$preprocessed_data
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$preprocessed_data <- prep_out$prep
      }
    }
  }

  # Standardize
  if (train_params$standardize != FALSE && !exists("prep_out")) {
    for (i in iters) {
      if (i == "split" && exists("df_list")) {
        prep_out <- .prep_data(
          train = df_list$split$train, test = df_list$split$test, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        df_list$split <- prep_out[!names(prep_out) == "prep"]
      } else if (startsWith(i, "fold") && exists("df_list")) {
        prep_out <- .prep_data(
          train = df_list$cv[[i]]$train, test = df_list$cv[[i]]$test, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        df_list$cv[[i]] <- prep_out[!names(prep_out) == "prep"]
      } else {
        prep_out <- .prep_data(
          preprocessed_data = preprocessed_data, vars = vars,
          train_params = train_params, impute_params = impute_params
        )

        preprocessed_data <- prep_out$preprocessed_data
      }
    }
  }

  # Create kwargs
  if (exists("df_list")) {
    kwargs <- list(
      df_list = df_list,
      formula = final_output$configs$formula,
      model_params = model_params,
      vars = vars,
      train_params = train_params,
      col_levels = col_levels,
      class_summary = final_output$class_summary,
      save_mods = save$models,
      met_df = final_output$metrics
    )
  }

  # Iterate to obtain validation metrics, training models, and final model for each algo
  for (model in models) {
    if (exists("kwargs")) {
      if (is.null(parallel_configs$n_cores) || parallel_configs$n_cores <= 1) {
        kwargs$iters <- iters[!iters == "final"]
        kwargs$model <- model
        train_out <- do.call(.train, kwargs)
      } else {
        kwargs$model <- model
        train_out <- .train_par(kwargs, parallel_configs, iters[!iters == "final"])
      }


      # Add metrics information and model information
      if ("split" %in% iters) {
        final_output$metrics[[model]]$split <- train_out$metrics$split
        train_out$metrics <- train_out$metrics[!names(train_out$metrics) == "split"]
      }

      if (!is.null(train_params$n_folds)) {
        cv_df <- .merge_df(
          iters[!iters %in% c("split", "final")],
          train_out$metrics$cv,
          final_output$metrics[[model]]$cv
        )

        final_output$metrics[[model]]$cv <- .get_desc(cv_df, train_params$n_folds)
      }

      if ("models" %in% names(train_out)) final_output$models[[model]] <- train_out$models
    }

    # Generate final model
    if ("final" %in% iters) {
      # Generate model depending on chosen models
      final_output$models[[model]]$final <- .generate_model(
        model = model,
        formula = final_output$configs$formula,
        vars = vars,
        data = preprocessed_data,
        add_args = model_params$mod_args,
        random_seed = train_params$random_seed
      )
    }
  }

  # Save data
  if (save$data == TRUE) {
    if (exists("kwargs")) final_output$data_partitions$dataframes <- df_list

    if ("final" %in% iters) final_output$data_partitions$dataframes$preprocessed_data <- preprocessed_data
  }

  # Make list a vswift class
  class(final_output) <- "vswift"
  return(final_output)
}
