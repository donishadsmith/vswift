# A list mapping default keys to certain parameters; Lazy evaluation
.DEFAULT_KEYS <- substitute({
  list(
    "model_params" = list(
      "map_args" = NULL,
      "threshold" = NULL,
      "rule" = "min",
      "final_model" = FALSE,
      "verbose" = TRUE
    ),
    "train_params" = .train_params_keys(call),
    "impute_params" = list(
      "method" = NULL,
      "args" = NULL
    ),
    "save" = list(
      "models" = FALSE,
      "data" = FALSE
    ),
    "parallel_configs" = list(
      "n_cores" = NULL,
      "future.seed" = NULL
    )
  )
})

# Helper function to return train keys depending on function call
.train_params_keys <- function(call) {
  keys <- list("split" = NULL, "n_folds" = NULL, "stratified" = FALSE, "random_seed" = NULL)

  if (call == "classCV") keys <- c(keys, list("standardize" = FALSE, "remove_obs" = FALSE))

  return(keys)
}

# A list mapping parameters to certain types; The %s_params parameters not included since their class is checked
# by .append_param_keys
.PARAM_TYPES <- list(
  primary = list(
    data = c("data.frame"),
    formula = c("formula", "NULL"),
    target = c("character", "numeric", "integer", "NULL"),
    predictors = c("character", "numeric", "integer", "NULL"),
    models = c("character"),
    model_params = c("list"),
    train_params = c("list"),
    impute_params = c("list"),
    parallel_configs = c("list"),
    save = c("list"),
    create_data = c("logical") # Only in genFolds
  ),
  secondary = list(
    map_args = c("list", "NULL"),
    threshold = c("numeric", "NULL"),
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
    future.seed = c("numeric", "NULL")
  )
)

# A list mapping models to their proper names
.MODEL_LIST <- list(
  "lda" = "Linear Discriminant Analysis",
  "qda" = "Quadratic Discriminant Analysis",
  "svm" = "Support Vector Machine",
  "nnet" = "Neural Network",
  "decisiontree" = "Decision Tree",
  "randomforest" = "Random Forest",
  "xgboost" = "Extreme Gradient Boosting",
  "logistic" = "Unegularized Logistic Regression",
  "regularized_logistic" = "Regularized Logistic Regression",
  "regularized_multinomial" = "Regularized Multinomial Logistic Regression",
  "multinom" = "Unregularized Multinomial Logistic Regression",
  "knn" = "K-Nearest Neighbors", "naivebayes" = "Naive Bayes"
)


# List of valid arguments for each model type
.GLMNET_ARGS <- c(
  "alpha", "lambda", "penalty.factor", "maxit", "thresh", "nfolds"
)

# List of valid arguments for each model type
.VALID_ARGS <- list(
  "model" = list(
    "lda" = c("prior", "method", "nu", "tol"),
    "qda" = c("prior", "method", "nu"),
    "logistic" = c("weights", "singular.ok", "maxit"),
    "regularized_logistic" = .GLMNET_ARGS,
    "svm" = c(
      "kernel", "degree", "gamma", "cost", "nu", "class.weights", "shrinking",
      "epsilon", "tolerance", "cachesize"
    ),
    "naivebayes" = c(
      "prior", "laplace", "usekernel", "usepoisson"
    ),
    "nnet" = c(
      "size", "rang", "decay", "maxit", "softmax", "entropy", "abstol", "reltol", "Hess",
      "skip"
    ),
    "knn" = c("kmax", "ks", "distance", "kernel"),
    "decisiontree" = c("method", "parms", "control", "cost"),
    "randomforest" = c(
      "classwt", "ntree", "mtry", "nodesize", "importance", "localImp",
      "nPerm", "proximity", "keep.forest", "norm.votes"
    ),
    "multinom" = c("Hess"),
    "regularized_multinomial" = .GLMNET_ARGS,
    "xgboost" = c(
      "params", "nrounds", "print_every_n", "feval", "verbose",
      "early_stopping_rounds", "obj", "save_period", "save_name"
    )
  ),
  "imputation" = list(
    "impute_bag" = c("trees", "seed_val"),
    "impute_knn" = c("neighbors")
  )
)
