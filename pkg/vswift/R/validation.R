# Helper function for classCV that performs validation
.validation <- function(id, train, test, model, formula, model_params, vars, remove_obs, col_levels, classes, keys,
                        met_df, random_seed, save) {
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
  train_mod <- .generate_model(
    model = model, data = train, formula = formula, vars = vars,
    add_args = model_params$map_args[[model]], random_seed = random_seed
  )

  # Remove observations where certain categorical levels in the predictors were not seen during training
  if (remove_obs && !is.null(col_levels)) test <- .remove_obs(train, test, col_levels, id)$test

  thresh <- if (model %in% c("logistic", "xgboost")) model_params$logistic_threshold else NULL
  obj <- if (model == "xgboost") model_params$map_args$xgboost$params$objective else NULL
  n_classes <- length(classes)

  # Get predictions for train and test set
  vec <- .prediction(
    id, model, train_mod, vars, list("train" = train, "test" = test),
    thresh, obj, n_classes
  )

  # Delete
  rm(train, test)
  gc()

  # Convert labels back
  if (model %in% c("logistic", "xgboost")) {
    for (name in names(vec$ground)) {
      if (name == "train") {
        vec$ground[[name]] <- .convert_keys(vec$ground[[name]], keys, "decode")
      }
      vec$pred[[name]] <- .convert_keys(vec$pred[[name]], keys, "decode")
    }
  }

  met_df <- .populate_metrics_df(id, classes, vec, met_df)

  if (save) {
    return(list("met_df" = met_df, "train_mod" = train_mod))
  } else {
    return(list("met_df" = met_df))
  }
}

# Helper function to convert keys
.convert_keys <- function(target_vector, keys, direction) {
  if (direction == "encode") {
    labels <- sapply(target_vector, function(x) keys[[x]])
  } else {
    converted_keys <- as.list(names(keys))
    names(converted_keys) <- as.character(as.vector(unlist(keys)))
    labels <- sapply(target_vector, function(x) converted_keys[[as.character(x)]])
  }
  return(labels)
}

# Helper function for to remove observations in test set with factors in predictors not observed during train
.remove_obs <- function(train, test, col_levels, id) {
  # Iterate over columns and check for the factors that exist in test set but not the train set
  for (col in names(col_levels)) {
    delete_rows <- which(!test[, col] %in% train[, col])
    obs <- row.names(test)[delete_rows]
    if (length(obs) > 0) {
      warning(sprintf(
        "for predictor `%s` in `%s` data partition has at least one class the model has not trained on\n  these observations will be temorarily removed: %s",
        col, id, paste(obs, collapse = ",")
      ))
      test <- test[-delete_rows, ]
    }
  }
  return(list("test" = test))
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
