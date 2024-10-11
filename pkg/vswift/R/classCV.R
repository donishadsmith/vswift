#' Perform Train-Test Split and/or K-Fold Cross-Validation with optional stratified sampling for classification data
#'
#' @name classCV
#' @description performs a train-test split and/or k-fold cross validation on classification data using various
#' classification algorithms.
#'
#' @param data A data frame containing the dataset. Default = \code{NULL}
#' @param formula A formula specifying the model to use. Cannot be used when \code{target} or \code{target} is
#'                specified. Default = \code{NULL}
#' @param target The target variable's numerical column index or name in the data frame. Cannot be used when
#'               \code{formula} is specified. Default = \code{NULL}.
#' @param predictors A vector of numerical indices or names for the predictors in the data frame. Used along with
#'                   \code{target}. If not specified when \code{target} is specified, all variables except the response
#'                   variable will be used a predictors. Cannot be used when \code{formula} is specified.
#'                   Default = \code{NULL}.
#' @param models A character character or character vector indicating the classification algorithm to use. Available
#'               options: \code{"lda"} (Linear Discriminant Analysis), \code{"qda"} (Quadratic Discriminant Analysis),
#'               \code{"logistic"} (Logistic Regression), \code{"svm"} (Support Vector Machines),
#'               \code{"naivebayes"} (Naive Bayes), \code{"ann"} (Artificial Neural Network), \code{"knn"}
#'               (K-Nearest Neighbors), \code{"decisiontree"} (Decision Tree), \code{"randomforest"} (Random Forest),
#'               \code{"multinom"} (Multinomial Logistic Regression), \code{"gbm"} (Gradient Boosting Machine).
#'                \itemize{
#'                \item For \code{"knn"}, the optimal k will be used unless specified with \code{ks}.
#'                \item For \code{"ann"}, \code{size} must be specified as an additional argument.
#'                }
#' @param model_params A list that can contain the following parameters:
#'                     \itemize{
#'                     \item \code{"map_args"}: A list of named sub-lists to use if more than one model is specified in
#'                     \code{models}. Each sub-list corresponds to a model specified in the \code{models}
#'                     parameter, and contains the parameters to be passed to the respective model.
#'                     Default = \code{NULL}. Please refer to "Additional Model Parameters" section for accepted
#'                     arguments.
#'                     \item \code{"logistic_threshold"}, A number between 0 and 1 indicating representing the
#'                     decision boundary for logistic regression. This parameter determines if an observation is
#'                     assigned to the class coded as "1" if \code{P(Class = 1 | Features) > logistic_threshold} or to
#'                     the class coded as "0" if \code{P(Class = 1 | Features) <= logistic_threshold}.
#'                     Default = \code{0.5}.
#'                     \item \code{"final_model"}: A logical value to use all complete observations in the input data
#'                     for model training. Default = \code{FALSE}.
#'                     }
#' @param train_params A list that can contain the following parameters:
#'                     \itemize{
#'                     \item \code{split}: A number from 0 to 1 for the proportion of data to use for the
#'                     training set, leaving the rest for the test set. If not specified, train-test splitting will not
#'                     be done. Note, this parameter is used to perform train-test splitting, which is separate
#'                     from cross-validation. Can be set to NULL, to not perform train-test splitting.
#'                     Default = \code{NULL}.
#'                     \item \code{n_folds}: An integer greater than 2 that indicates the number of folds to use for
#'                     k-fold cross validation (CV). Note, k-fold CV is performed separately from train-test splitting.
#'                     Can be set to NULL, to not perform k-fold CV. Default = \code{NULL}
#'                     \item \code{stratified}: A logical value indicating if stratified sampling should be used.
#'                     Default = \code{FALSE}.
#'                     \item \code{random_seed} A numerical value for the random seed to ensure random splitting and
#'                     any stochastic model results are reproducible. Default = \code{NULL}.
#'                     \item \code{standardize}: A logical value or numerical vector. If \code{TRUE}, all columns
#'                     except the target, that are numeric, will be standardized. Standardization is done by
#'                     calculating the column means and standard deviation for the training subset of the train-test
#'                     split or each fold. Then each column of the training and test/validation set are standardized
#'                     using the same mean and standard deviation. This is done to prevent data leakage. To specify the
#'                     columns to be standardized, create a numerical or character vector consisting of the column
#'                     indices or names to be standardized.
#'                     \item \code{remove_obs}: A logical value to remove observations with categorical predictors
#'                     from the test/validation set that have not been observed during model training. Some algorithms
#'                     may produce an error if this occurs; thus, this parameter is intended to prevent this error.
#'                     Default = \code{FALSE}.
#'                     }
#' @param impute_params A list that can contain the following parameters to perform imputation on missing
#'                      predictors/features. During imputation, the targets are not included in the training or
#'                      test/validation set. Prior to imputation, train-test splitting, or k-fold CV generation,
#'                      any rows with missing data for the targets are dropped completely. Additionally, imputation
#'                      models are generated for each training data (one model for the training subset of the train-test
#'                      split, as well as one model for the training subset of each k-fold; so the imputation model for
#'                      fold 1 is not the same model used for fold 2) and the same model is used for both the training
#'                      and test/validation data. This is done to minimize data leakage. Also, standardization of
#'                      numerical columns will be done automatically regardless if \code{train_params$standardize} is
#'                      set to \code{False}. The recipe package is used for imputation and the following parameters can
#'                      be used:
#'                      \itemize{
#'                      \item \code{method}: A character indicating the imputation method to use. Options include
#'                      \code{"bag_impute"} (Bagged Trees Imputation) and \code{"knn_impute"} (KNN Imputation).
#'                      \item \code{args}: A list specifying an additional argument for the imputation method.
#'                      Below are the additional arguments available for each imputation option.
#'                      \itemize{
#'                      \item \code{"bag_impute"}: \code{neighbors}
#'                      \item \code{"knn_impute"}: \code{trees}, \code{seed_val}
#'                      }
#'                      For specific information about each parameter, please refer to the recipes documentation.
#'                      Default = \code{NULL}.
#'                      }
#' @param save A list that can contain the following parameters:
#'             \itemize{
#'             \item \code{models}: A logical value to save training models (includes imputation models) used for
#'             train_test splitting or k-fold CV
#'             validation. Default = \code{FALSE}.
#'             \item \code{data}: A logical value to save all training and test/validation sets during train-test
#'             splitting and/or k-fold CV. Default = \code{FALSE}.
#'             }
#' @param parallel_configs A list that can contain the following parameters:
#'                         \itemize{
#'                         \item \code{n_cores}: A numerical value specifying the number of cores to use for parallel
#'                         processing. Default = \code{NULL}.
#'                         \item \code{future.seed}: A numerical value indicating the seed to use when calling future
#'                         for parallel processing.
#'                         }
#' @param ... Additional arguments specific to the chosen classification algorithm. To be used as an alternate to
#'            specifying model specific parameters using \code{map_args} in \code{model_params} when only a single
#'            model is specified in \code{models}. If multiple models are specified, then \code{map_args} must be
#'            used. Please refer to the corresponding algorithm's documentation for additional arguments and their
#'            descriptions.
#'
#' @section Additional Model Parameters:
#'   Each option of \code{models} accepts additional arguments specific to the classification algorithm.
#'   For additional information about these arguments, refer to the documentation in the original packages. Information
#'   on the external package functions used for each models can be found in the "Package Dependencies" section.
#'   The available arguments for each \code{models} are:
#'   \itemize{
#'    \item \code{"lda"}: \code{prior}, \code{method}, \code{nu}, \code{tol}
#'    \item \code{"qda"}: \code{prior}, \code{method}, \code{nu}
#'    \item \code{"logistic"}: \code{weights}, \code{singular.ok}, \code{maxit}
#'    \item \code{"svm"}: \code{kernel}, \code{degree}, \code{gamma}, \code{cost}, \code{nu}, \code{class.weights},
#'                        \code{shrinking}, \code{epsilon}, \code{tolerance}, \code{cachesize}
#'    \item \code{"naivebayes"}: \code{prior}, \code{laplace}, \code{usekernel}, \code{bw}, \code{kernel},
#'                               \code{adjust}, \code{weights}, \code{give.Rkern}, \code{subdensity}, \code{from},
#'                               \code{to}, \code{cut}
#'    \item \code{"ann"}: \code{size}, \code{rang}, \code{decay}, \code{maxit}, \code{softmax},
#'                        \code{entropy}, \code{abstol}, \code{reltol}, \code{Hess}
#'    \item \code{"knn"}: \code{kmax}, \code{ks}, \code{distance}, \code{kernel}
#'    \item \code{"decisiontree"}: \code{weights}, \code{method},\code{parms}, \code{control}, \code{cost}
#'    \item \code{"randomforest"}: \code{weights}, \code{ntree}, \code{mtry}, \code{nodesize}, \code{importance},
#'                                 \code{localImp}, \code{nPerm}, \code{proximity}, \code{keep.forest},
#'                                 \code{norm.votes}
#'    \item \code{"multinom"}: \code{weights}, \code{Hess}
#'    \item \code{"gbm"}: \code{params}, \code{nrounds}, \code{print_every_n}, \code{feval}, \code{verbose},
#'                        \code{early_stopping_rounds}, \code{obj}, \code{save_period}, \code{save_name}
#'   }
#'
#' @section Package Dependencies:
#'   Each option of \code{models} uses the following function from the specified packages:
#'   \itemize{
#'    \item \code{"lda"}: \code{lda()} from MASS package
#'    \item \code{"qda"}: \code{qda()} from MASS package
#'    \item \code{"logistic"}: \code{glm()} from base package with \code{family = "binomial"}
#'    \item \code{"svm"}: \code{svm()} from e1071 package
#'    \item \code{"naivebayes"}: \code{naive_bayes()} from naivebayes package
#'    \item \code{"ann"}: \code{nnet()} from nnet package
#'    \item \code{"knn"}: \code{train.kknn()} from kknn package
#'    \item \code{"decisiontree"}: \code{rpart()} from rpart package
#'    \item \code{"randomforest"}: \code{randomForest()} from randomForest package
#'    \item \code{"multinom"}: \code{multinom()} from nnet package
#'    \item \code{"gbm"}: \code{xgb.train()} from xgboost package
#'   }
#'
#' @return A list containing the results of train-test splitting and/or k-fold cross-validation (if specified),
#'         performance metrics, information on the class distribution in the training, test sets, folds
#'         (if applicable), saved models (if specified), saved datasets (if specified), a final model
#'         (if specified).
#'
#' @seealso \code{\link{print.vswift}}, \code{\link{plot.vswift}}
#'
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#' result <- classCV(data = iris,
#'                   target = "Species",
#'                   models = "lda",
#'                   train_params = list(split = 0.8)
#'                   )
#'
#' # Print parameters and metrics
#' result
#'
#' # Perform 5-fold cross-validation using Gradient Boosted Model
#' # w/ additional parameters: params & nrounds
#' result <- classCV(data = iris,
#'                   formula = Species ~ .,
#'                   models = "gbm",
#'                   train_params = list(n_folds = 5, random_seed = 50),
#'                   params = list(objective = "multi:softprob",
#'                                 num_class = 3,
#'                                 eta = 0.3,
#'                                 max_depth = 6),
#'                   nrounds = 10
#'                   )
#'
#' # Print parameters and metrics
#' result
#'
#'
#' # Perform 5-fold cross-validation a train-test split w/multiple models
#' args <- list("knn" = list(ks = 5), "ann" = list(size = 20))
#' result <- classCV(data = iris,
#'                   target = 5,
#'                   predictors = c(1:3),
#'                   models = c("decisiontree","knn", "ann","svm"),
#'                   model_params = list(map_args = args),
#'                   train_params = list(n_folds = 5,
#'                                          stratified = TRUE,
#'                                          random_seed = 50)
#'                   )
#'
#' # Print parameters and metrics
#' result
#'
#' @author Donisha Smith
#'
#' @export
classCV <- function(data,
                    formula = NULL,
                    target = NULL,
                    predictors = NULL,
                    models,
                    model_params = list("map_args" = NULL, "logistic_threshold" = 0.5, "final_model" = FALSE),
                    train_params = list("split" = NULL, "n_folds" = NULL, "stratified" = FALSE,
                                        "random_seed" = NULL, "standardize" = FALSE, "remove_obs" = FALSE),
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
  .error_handling(data = data, formula = formula, target = target, predictors = predictors, models = models,
                  model_params = model_params, train_params = train_params, impute_params = impute_params,
                  save = save, parallel_configs = parallel_configs, call = "classCV")

  # Get character form of target and predictor variables
  vars <- .get_var_names(formula, target, predictors, data)

  # Remove missing data if no imputation specified and remove rows with missing target variables.
  if (is.null(impute_params$method)) {
    preprocessed_data <- .remove_missing_data(data)
    miss_data <- FALSE
  } else {
    preprocessed_data <- .remove_missing_target(data, target)
    # Check if removing missing target variables removes all missing data
    miss_data <- .check_if_missing(preprocessed_data)
  }

  # Ensure target is factored and get all levels of character columns obtained if svm in models
  factored <- .convert_to_factor(preprocessed_data, vars$target, models, train_params)
  preprocessed_data <- factored$data
  col_levels <- factored$col_levels

  missing_n <- nrow(data) - nrow(preprocessed_data)
  # Delete data
  rm(data, factored); gc()

  # Store information
  final_output <- .store_parameters(formula, missing_n, preprocessed_data, vars, models, model_params, train_params,
                                    impute_params, save, parallel_configs)

  # Create class dictionary
  if (any(models %in% c("logistic", "gbm"))) {
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
  if (!is.null(impute_params$method) && miss_data == TRUE) {
    for (i in iters) {
      if (i == "split" && exists("df_list")) {
        prep_out <- .prep_data(train = df_list$split$train, test = df_list$split$test, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        df_list$split <- prep_out[!names(prep_out) == "prep"]
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$split <- prep_out$prep

      } else if (startsWith(i, "fold") && exists("df_list")) {
        prep_out <- .prep_data(train = df_list$cv[[i]]$train, test = df_list$cv[[i]]$test, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        df_list$cv[[i]] <- prep_out[!names(prep_out) == "prep"]
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$cv[[i]] <- prep_out$prep

      } else {
        prep_out <- .prep_data(preprocessed_data = preprocessed_data, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        preprocessed_data <- prep_out$preprocessed_data
        if ("prep" %in% names(prep_out) & save$models == TRUE) final_output$imputation$preprocessed_data <- prep_out$prep
      }
    }
  }

  # Standardize
  if (train_params$standardize != FALSE && !exists("prep_out")) {
    for (i in iters) {
      if (i == "split" && exists("df_list")) {
        prep_out <- .prep_data(train = df_list$split$train, test = df_list$split$test, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        df_list$split <- prep_out[!names(prep_out) == "prep"]
      } else if (startsWith(i, "fold") && exists("df_list")) {
        prep_out <- .prep_data(train = df_list$cv[[i]]$train, test = df_list$cv[[i]]$test, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        df_list$cv[[i]] <- prep_out[!names(prep_out) == "prep"]
      } else {
        prep_out <- .prep_data(preprocessed_data = preprocessed_data, vars = vars,
                               train_params = train_params, impute_params = impute_params)

        preprocessed_data <- prep_out$preprocessed_data
      }
    }
  }

  # Create kwargs
  if (exists("df_list")) {
    kwargs <- list(df_list = df_list,
                   formula = final_output$configs$formula,
                   model_params = model_params,
                   vars = vars,
                   train_params = train_params,
                   col_levels = col_levels,
                   class_summary = final_output$class_summary,
                   save_mods = save$models,
                   met_df = final_output$metrics)
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
        cv_df <- .merge_df(iters[!iters %in% c("split","final")],
                           train_out$metrics$cv,
                           final_output$metrics[[model]]$cv)

        final_output$metrics[[model]]$cv <- .get_desc(cv_df, train_params$n_folds)
      }

      if ("models" %in% names(train_out)) final_output$models[[model]] <- train_out$models

    }

    # Generate final model
    if ("final" %in% iters) {
      # Generate model depending on chosen models
      final_output$models[[model]]$final <- .generate_model(model = model,
                                                            formula = final_output$configs$formula,
                                                            vars = vars,
                                                            data = preprocessed_data,
                                                            add_args = model_params$mod_args,
                                                            random_seed = train_params$random_seed)
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
