# Sampling function used by classCV and genFolds
.sampling <- function(data, train_params, target, final_output) {
  # Base args
  base_args <- list(N = nrow(data), random_seed = train_params$random_seed)

  if (isTRUE(train_params$stratified)) {
    # Create args list
    strat_args <- list(
      classes = final_output$class_summary$classes,
      class_indxs = final_output$class_summary$indices,
      class_props = final_output$class_summary$proportions
    )

    strat_args <- c(base_args, strat_args)
    # Get stratified indices
    if (!is.null(train_params$split)) {
      strat_args$split <- train_params$split
      final_output$data_partitions$indices$split <- do.call(.stratified_split, strat_args)
      # Get proportions of classes in the stratified indices
      final_output$data_partitions$proportions$split <- .get_proportions(
        data[, target],
        final_output$data_partitions$indices$split
      )
    }

    if (!is.null(train_params$n_folds)) {
      # Remove split arg
      strat_args <- strat_args[!names(strat_args) == "split"]
      strat_args$n_folds <- train_params$n_folds
      final_output$data_partitions$indices$cv <- do.call(.stratified_cv, strat_args)
      # Get proportions of classes in the stratified indices
      final_output$data_partitions$proportions$cv <- .get_proportions(
        data[, target],
        final_output$data_partitions$indices$cv
      )
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
