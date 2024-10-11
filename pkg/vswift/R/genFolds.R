#' Create split datasets and/or folds with optional stratification
#'
#' @name genFolds
#' @description Standalone function generates train-test split datasets and/or k-fold cross-validation folds, with the
#'              option to perform stratified sampling based on class distribution.
#'
#' @param data A data frame.
#' @param target A numerical index or character name for the target variable. Only needs to be specified if
#'               \code{stratified = TRUE}. Default = \code{NULL}.
#' @param train_params A list that can contain the following parameters:
#'                     \itemize{
#'                     \item \code{"split"}: A number from 0 to 1 for the proportion of data to use for the
#'                     training set, leaving the rest for the test set. If not specified, train-test splitting will not
#'                     be done. Note, this parameter is used to perform train-test splitting, which is separate
#'                     from cross-validation. Can be set to NULL, to not perform train-test splitting.
#'                     Default = \code{NULL}.
#'                     \item \code{"n_folds"}: An integer greater than 2 that indicates the number of folds to use for
#'                     k-fold cross validation (CV). Note, k-fold CV is performed separately from train-test splitting.
#'                     Can be set to NULL, to not perform k-fold CV. Default = \code{NULL}
#'                     \item \code{"stratified"}: A logical value indicating if stratified sampling should be used.
#'                     Default = \code{FALSE}.
#'                     \item \code{"random_seed"} A numerical value for the random seed to ensure random splitting
#'                     are reproducible. Default = \code{NULL}.
#'                     }
#' @param create_data A logical value indicating whether to create all training and test/validation data frames.
#'                    Default = \code{FALSE}.
#' @return A list containing the indices for train-test splitting and/or k-fold cross-validation, with information on
#'         the class distribution in the training, test sets, and folds (if applicable) as well as the generated split
#'         datasets and folds based on the indices.
#' @examples
#' # Load example dataset
#'
#' data(iris)
#'
#' # Obtain indices for 80% training/test split and 5-fold CV
#'
#' output <- genFolds(data = iris, target = "Species", train_params = list(split = 0.8, n_folds = 5))
#'
#' @author Donisha Smith
#' @export
genFolds <- function(data,
                     target,
                     train_params = list(split = NULL, n_folds = NULL, stratified = FALSE, random_seed = NULL),
                     create_data = FALSE) {

  # Append train_params
  train_params <- .append_keys("train_params", train_params, call = "genFolds")
  # Check validity of inputs
  .error_handling(data = data, target = target, train_params = train_params, create_data = create_data, call = "genFolds")

  # Initialize final output list
  final_output <- list("configs" = train_params)
  final_output <- c(final_output, .append_output(data[,target], train_params$stratified))
  final_output$data_partitions <- list()

  # Perform sampling
  final_output <- .sampling(data, train_params, target, final_output)

  # Get data partitions
  if (create_data == TRUE) final_output$data_partitions$dataframes <- .partition(data, final_output$data_partitions$indices)

  return(final_output)
}

# Sampling function used by classCV and genFolds
.sampling <- function(data, train_params, target, final_output){
  # Base args
  base_args <- list(N = nrow(data), random_seed = train_params$random_seed)

  if (train_params$stratified == TRUE) {
    # Create args list
    strat_args <- list(classes = final_output$class_summary$classes,
                       class_indxs = final_output$class_summary$indices,
                       class_props = final_output$class_summary$proportions)

    strat_args <- c(base_args, strat_args)
    # Get stratified indices
    if (!is.null(train_params$split)) {
      strat_args$split <- train_params$split
      final_output$data_partitions$indices$split <- do.call(.stratified_split, strat_args)
      # Get proportions of classes in the stratified indices
      final_output$data_partitions$proportions$split <- .get_proportions(data[,target],
                                                                         final_output$data_partitions$indices$split)
    }

    if (!is.null(train_params$n_folds)) {
      # Remove split arg
      strat_args <- strat_args[!names(strat_args) == "split"]
      strat_args$n_folds <- train_params$n_folds
      final_output$data_partitions$indices$cv <- do.call(.stratified_cv, strat_args)
      # Get proportions of classes in the stratified indices
      final_output$data_partitions$proportions$cv <- .get_proportions(data[,target],
                                                                      final_output$data_partitions$indices$cv)
    }

  } else {
    # Non-stratified sampling
    if (!is.null(train_params$split)) {
      base_args$split <- train_params$split
      final_output$data_partitions$indices$split <- do.call(.split, base_args)
    }

    if (!is.null(train_params$n_folds)) {
      # Remove split arg
      base_args <- base_args[!names(base_args) == "split"]
      base_args$n_folds <- train_params$n_folds
      final_output$data_partitions$indices$cv <- do.call(.cv, base_args)
    }
  }

  return(final_output)
}
