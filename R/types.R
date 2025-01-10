# Parameter type list
.param_types <- list(
  primary = list(
    data = c("data.frame"),
    formula = c("formula", "NULL"),
    target = c("character", "numeric", "integer", "NULL"),
    predictors = c("character", "numeric", "integer", "NULL"),
    models = c("character"),
    model_params = c("list"),
    train_params = c("list"),
    impute_params = c("list"),
    save = c("list"),
    parallel_configs = c("list")
  ),
  secondary = list(
    map_args = c("list", "NULL"),
    logistic_threshold = c("numeric", "NULL"),
    rule = c("character", "NULL"),
    final_model = c("logical"),
    verbose = c("logical", "NULL"),
    split = c("numeric", "NULL"),
    n_folds = c("numeric", "NULL"),
    stratified = c("logical"),
    random_seed = c("numeric", "NULL"),
    standardize = c("logical", "numeric", "integer", "character"),
    remove_obs = c("logical"),
    method = c("character", "NULL"),
    args = c("list", "NULL"),
    models = c("logical"),
    data = c("logical"),
    n_cores = c("numeric", "NULL"),
    future.seed = c("numeric", "NULL"),
    create_data = c("logical") # Only in genFolds
  )
)

# Code for checking the typing of parameters
.param_checker <- function(param, value) {
  msg <- "`%s` must be of the following classes: %s"
  types <- .param_types$primary[[param]]

  if (!inherits(value, types)) stop(sprintf(msg, param, paste(types, collapse = ", ")))

  if ("list" %in% types) {
    secondary_names <- names(value)
    for (name in secondary_names) {
      types <- .param_types$secondary[[name]]
      if (!inherits(value[[name]], types)) stop(sprintf(msg, name, paste(types, collapse = ", ")))
    }
  }
}
