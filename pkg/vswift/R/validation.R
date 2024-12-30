# Helper function for classCV that performs validation
.validation <- function(id, train, test, model, formula, model_params, vars, remove_obs, col_levels, stratified,
                        classes, keys, met_df, random_seed, save) {
  # Ensure factored columns have same levels for svm
  if (model == "svm" && !is.null(col_levels)) {
    train[, names(col_levels)] <- data.frame(lapply(names(col_levels), function(col) factor(train[, col], levels = col_levels[[col]])))
    test[, names(col_levels)] <- data.frame(lapply(names(col_levels), function(col) factor(test[, col], levels = col_levels[[col]])))
  }

  # Convert to numerical
  if (model %in% c("logistic", "xgboost")) {
    train[, vars$target] <- .convert_keys(train[, vars$target], keys, "encode")
  }

  # Train model
  if (startsWith(model, "regularized")) {
    train_mod <- .regularized(
      id = id, model = model, data = train, vars = vars,
      add_args = model_params$map_args[[model]], random_seed = random_seed,
      stratified = if (is.null(stratified)) FALSE else stratified,
      rule = if (is.null(model_params$rule)) "min" else model_params$rule
    )

    # Get optimal lambda
    if ("optimal_lambda" %in% names(train_mod)) {
      optimal_lambda <- train_mod$optimal_lambda
      train_mod$optimal_lambda <- NULL
    }
  } else {
    train_mod <- .generate_model(
      model = model, data = train, formula = formula, vars = vars,
      add_args = model_params$map_args[[model]], random_seed = random_seed
    )
  }

  # Remove observations where certain categorical levels in the predictors were not seen during training
  if (remove_obs && !is.null(col_levels)) test <- .remove_obs(train, test, col_levels, id)$test

  thresh <- if (endsWith(model, "logistic") | model == "xgboost") model_params$logistic_threshold else NULL
  obj <- if (model == "xgboost") model_params$map_args$xgboost$params$objective else NULL
  n_classes <- length(classes)

  # Get predictions for train and test set
  vec <- .prediction(
    id, model, train_mod, vars, list("train" = train, "test" = test),
    thresh, obj, n_classes
  )

  # Convert labels back
  if (model %in% c("logistic", "xgboost")) {
    for (name in names(vec$ground)) {
      if (name == "train") {
        vec$ground[[name]] <- .convert_keys(vec$ground[[name]], keys, "decode")
      }
      vec$pred[[name]] <- .convert_keys(vec$pred[[name]], keys, "decode")
    }
  }

  metrics <- .populate_metrics_df(id, classes, vec, met_df)

  # Output
  out <- list("metrics" = metrics)

  if (save) out$train_mod <- train_mod

  if (exists("optimal_lambda")) out$optimal_lambda <- optimal_lambda

  return(out)
}

# Helper function for classCV to create model
.generate_model <- function(model, data, formula, vars = NULL, add_args = NULL, random_seed = NULL) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)

  if (model == "xgboost") {
    mod_args <- list()
  } else {
    mod_args <- list(formula = formula, data = data)
  }

  if (model == "logistic") mod_args$family <- "binomial"

  if (!is.null(add_args)) mod_args <- c(mod_args, add_args)

  # Prevent default internal scaling for models with the scale parameter
  if (!model %in% c("decisiontree", "xgboost", "logistic")) mod_args$scale <- FALSE

  switch(model,
    "lda" = {
      model <- do.call(MASS::lda, mod_args)
    },
    "qda" = {
      model <- do.call(MASS::qda, mod_args)
    },
    "logistic" = {
      model <- do.call(glm, mod_args)
    },
    "svm" = {
      model <- do.call(e1071::svm, mod_args)
    },
    "naivebayes" = {
      model <- do.call(naivebayes::naive_bayes, mod_args)
    },
    "nnet" = {
      model <- do.call(nnet::nnet.formula, mod_args)
    },
    "knn" = {
      model <- do.call(kknn::train.kknn, mod_args)
    },
    "decisiontree" = {
      model <- do.call(rpart::rpart, mod_args)
    },
    "randomforest" = {
      model <- do.call(randomForest::randomForest, mod_args)
    },
    "multinom" = {
      model <- do.call(nnet::multinom, mod_args)
    },
    "xgboost" = {
      mat_data <- data.matrix(data)
      mod_args$data <- xgboost::xgb.DMatrix(data = mat_data[, vars$predictors], label = mat_data[, vars$target])
      model <- do.call(xgboost::xgb.train, mod_args)
    }
  )

  return(model)
}

# Helper function to perform regularized logistic or multinomial regression
.regularized <- function(id, model, data, vars = NULL, add_args = NULL, random_seed = NULL, stratified = FALSE,
                         rule = "min", verbose = TRUE) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)

  mod_args <- list()
  cv_args <- list()
  cv_flag <- FALSE

  base_kwargs <- list()
  # Create x and y matrices
  base_kwargs$x <- model.matrix(~ . - 1, data = data[, vars$predictors])
  base_kwargs$y <- as.matrix(data[, vars$target])

  # Family
  base_kwargs$family <- ifelse(model == "regularized_logistic", "binomial", "multinomial")

  # Prevent default scaling
  base_kwargs$standardize <- FALSE

  # Classification measure
  base_kwargs$type.measure <- "class"

  # Append base_kwargs
  cv_args <- c(cv_args, base_kwargs)
  mod_args <- c(mod_args, base_kwargs)

  # Additional arguments
  if (!is.null(add_args)) {
    cv_args <- c(mod_args, add_args)
    mod_args <- c(mod_args, add_args[!add_args %in% c("nfolds", "lambda")])
  }

  # If lambda is NULL or greater then one, use CV to identify optimal lambda
  if (length(mod_args$lambda) == 0 || length(mod_args$lambda) > 1) {
    # Attempt to retain a similar stratification that is in the training sample if train_params$stratified is TRUE
    if (stratified) {
      class_info <- .get_class_info(data[, vars$target])
      nfolds <- if (inherits(cv_args$nfolds, "numeric")) cv_args$nfolds else 10
      cv_indxs <- .stratified_cv(
        names(class_info$proportions), class_info$indices, class_info$proportions, nrow(data),
        nfolds, random_seed, "nested cross-validation (glmnet)"
      )
      cv_args$n_folds <- NULL
      cv_args$foldid <- .get_foldid(cv_indxs, nrow(data))
    }

    # Perform internal cross validation on data; default n_folds is 10
    cv.fit <- do.call(glmnet::cv.glmnet, cv_args)

    # Select optimal lambda based on rule
    mod_args$lambda <- ifelse(rule == "min", cv.fit$lambda.min, cv.fit$lambda.1se)

    cv_flag <- TRUE

    # State optimal lambda
    if (verbose) {
      if (id != "Final Model") {
        id <- ifelse(id == "split", "Train-Test Split", paste("Fold", unlist(strsplit(id, split = "fold"))[2]))
      }
      num <- ifelse(!is.null(mod_args$nfolds), mod_args$nfolds, 10)
      msg <- sprintf(
        "Model: %s | Partition: %s | Optimal lambda: %.5f (nested %s-fold cross-validation using '%s' rule)",
        model, id, mod_args$lambda, num, rule
      )
      cat(msg, "\n")
    }
  } else {
    mod_args$lambda <- add_args$lambda
  }

  # Get model
  model <- do.call(glmnet::glmnet, mod_args)

  out <- list("model" = model)

  if (cv_flag) {
    out <- c(out, list("cv.fit" = cv.fit, "optimal_lambda" = mod_args$lambda))
  }

  return(out)
}

# Helper function for classCV to predict
.prediction <- function(id, mod, train_mod, vars, df_list, thresh, obj, n_classes) {
  # vec to store ground truth and predicted data
  vec <- list("ground" = list(), "pred" = list())
  # Only get predictions for training set if train-test split
  if (id == "split") {
    vec$ground$train <- as.vector(df_list$train[, vars$target])
  } else {
    df_list <- df_list[names(df_list) != "train"]
  }

  vec$ground$test <- as.vector(df_list$test[, vars$target])

  for (i in names(df_list)) {
    switch(mod,
      "lda" = {
        vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])$class
      },
      "qda" = {
        vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])$class
      },
      "logistic" = {
        vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]], type = "response")
        vec$pred[[i]] <- ifelse(vec$pred[[i]] >= thresh, 1, 0)
      },
      "naivebayes" = {
        vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]][, vars$predictors])
      },
      "nnet" = {
        vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]], type = "class")
      },
      "decisiontree" = {
        mat <- predict(train_mod, newdata = df_list[[i]])
        vec$pred[[i]] <- colnames(mat)[apply(mat, 1, which.max)]
      },
      "xgboost" = {
        mat <- data.matrix(df_list[[i]])
        xgb_mat <- xgboost::xgb.DMatrix(data = mat[, vars$predictors], label = mat[, vars$target])
        vec$pred[[i]] <- .handle_xgboost_predict(train_mod, xgb_mat, obj, thresh, n_classes)
      },
      "regularized_logistic" = {
        X <- model.matrix(~ . - 1, data = df_list[[i]][, vars$predictors])
        vec$pred[[i]] <- predict(train_mod$model, newx = X, s = train_mod$model$lambda, type = "class")
      },
      "regularized_multinomial" = {
        X <- model.matrix(~ . - 1, data = df_list[[i]][, vars$predictors])
        mat <- predict(train_mod$model, newx = X, s = train_mod$model$lambda, type = "response")
        vec$pred[[i]] <- colnames(mat)[apply(mat, 1, which.max)]
      },
      # Default for svm, knn, randomforest, and multinom
      vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])
    )
    vec$pred[[i]] <- as.vector(vec$pred[[i]])
  }

  return(vec)
}

# Handle different xgboost objective functions
.handle_xgboost_predict <- function(train_mod, xgb_mat, obj, thresh, n_classes) {
  # produces probability
  bin_prob <- c("reg:logistic", "binary:logistic")

  obj <- ifelse(obj %in% bin_prob, "binary_prob", obj)

  pred <- predict(train_mod, newdata = xgb_mat)

  # Special cases that need to be converted to labels
  switch(obj,
    "binary_prob" = {
      pred <- ifelse(pred >= thresh, 1, 0)
    },
    "binary:logitraw" = {
      pred <- sapply(pred, .logit2prob)
      pred <- ifelse(pred >= thresh, 1, 0)
    },
    "multi:softprob" = {
      pred <- max.col(matrix(pred, ncol = n_classes, byrow = TRUE)) - 1
    }
  )

  return(pred)
}

# Convert logit to probability
.logit2prob <- function(x) {
  prob <- exp(x) / (1 + exp(x))
  return(prob)
}
