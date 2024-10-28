# Function to perform validation of split and folds
.train <- function(iters, df_list, model, formula, model_params, vars, train_params, col_levels, class_summary,
                   save_mods, met_df) {

  met_list <- list()
  mod_list <- list()

  for (i in iters) {
    name <- ifelse(i == "split", i, "cv")
    if (name == "split") {
      train <- df_list[[i]]$train
      test <- df_list[[i]]$test
    } else {
      train <- df_list$cv[[i]]$train
      test <- df_list$cv[[i]]$test
    }
    val_out <- .validation(i, train, test, model, formula, model_params, vars, train_params$remove_obs, col_levels,
                           class_summary$classes, class_summary$keys, met_df[[model]][[name]], train_params$random_seed,
                           save_mods)
    if (name == "split") {
      met_list$split <- val_out$met_df
      if (save_mods == TRUE) mod_list$split <- val_out$train_mod
    } else {
      met_list$cv[[i]] <- val_out$met_df
      if (save_mods == TRUE) mod_list$cv[[i]] <- val_out$train_mod
    }
  }

  if (save_mods == FALSE) {
    return(list("metrics" = met_list))
  } else {
    return(list("metrics" = met_list, "models" = mod_list))
  }
}

# Function to perform validation of split and folds using parallel processing while combining outputs
#' @importFrom future plan multisession sequential
#' @importFrom future.apply future_lapply
.train_par <- function(kwargs, parallel_configs, iters) {
  plan(multisession, workers = parallel_configs$n_cores)
  par_out <- future_lapply(iters, function(i){
    kwargs$iters <- i
    val_out <- do.call(.train, kwargs)
  }, future.seed = if (!is.null(parallel_configs$future.seed)) parallel_configs$future.seed else TRUE)

  # Close the background workers
  plan(sequential)

  par_unnest <- .unnest(par_out, iters, kwargs$save_mods)
  return(par_unnest)
}

# Unnest parallel list
.unnest <- function(par_list, iters, saved_mods = NULL) {
  targets <- c("metrics")
  metrics <- list()

  if (saved_mods == TRUE) {
    targets <- c("metrics", "models")
    models <- list()
  }

  for (target in targets) {
    for (i in seq_along(iters)) {
      if (target == "metrics") {
        if (iters[i] == "split") {
          metrics$split <- par_list[[i]]$metrics$split
        } else {
          metrics$cv <- c(metrics$cv, par_list[[i]]$metrics$cv)
        }

      } else {
        if (iters[i] == "split") {
          models$split <- par_list[[i]]$models$split
        } else {
          models$cv <- c(models$cv, par_list[[i]]$models$cv)
        }
      }
    }
  }

  if (saved_mods == TRUE) {
    return(list("metrics" = metrics, "models" = models))
  } else {
    return(list("metrics" = metrics))
  }

}
# Helper function for classCV that performs validation
.validation <- function(id, train, test, model, formula, model_params, vars, remove_obs, col_levels, classes, keys,
                        met_df, random_seed, save) {

  # Ensure factored columns have same levels for svm
  if (model == "svm" && !is.null(col_levels)) {
    train[,names(col_levels)] <- data.frame(lapply(names(col_levels), function(col) factor(train[,col], levels = col_levels[[col]])))
    test[,names(col_levels)] <- data.frame(lapply(names(col_levels), function(col) factor(test[,col], levels = col_levels[[col]])))
  }

  # Convert to numerical
  if (model %in% c("logistic", "gbm")) {
    train[,vars$target] <- .convert_keys(train[,vars$target], keys, "encode")
  }
  # Train model
  train_mod <- .generate_model(model = model, data = train, formula = formula, vars = vars,
                               add_args = model_params$map_args[[model]], random_seed = random_seed)

  # Remove observations where certain categorical levels in the predictors were not seen during training
  if (remove_obs && !is.null(col_levels)) test <- .remove_obs(train, test, col_levels, id)$test

  thresh <- if (model %in% c("logistic", "gbm")) model_params$logistic_threshold else NULL
  obj <- if (model == "gbm") model_params$map_args$gbm$params$objective else NULL
  n_classes <- length(classes)

  # Get predictions for train and test set
  vec <- .prediction(id, model, train_mod, vars, list("train" = train, "test" = test),
                     thresh, obj, n_classes)

  # Delete
  rm(train, test); gc()

  # Convert labels back
  if (model %in% c("logistic", "gbm")) {
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
    names(converted_keys) <-  as.character(as.vector(unlist(keys)))
    labels <- sapply(target_vector, function(x) converted_keys[[as.character(x)]])
  }
  return(labels)
}

#Helper function for to remove observations in test set with factors in predictors not observed during train
.remove_obs <- function(train, test, col_levels, id) {
  # Iterate over columns and check for the factors that exist in test set but not the train set
  for (col in names(col_levels)) {
    delete_rows <- which(!test[,col] %in% train[,col])
    obs <- row.names(test)[delete_rows]
    if (length(obs) > 0) {
      warning(sprintf("for predictor `%s` in `%s` data partition has at least one class the model has not trained on\n  these observations will be temorarily removed: %s",
                      col, id, paste(obs, collapse = ",")))
      test <- test[-delete_rows,]
    }
  }
  return(list("test" = test))
}

# Helper function for classCV to create model
#' @importFrom MASS lda qda
#' @importFrom stats glm
#' @importFrom naivebayes naive_bayes
#' @importFrom e1071 svm
#' @importFrom nnet nnet.formula nnet multinom
#' @importFrom kknn train.kknn
#' @importFrom rpart rpart
#' @importFrom randomForest randomForest
#' @importFrom xgboost xgb.DMatrix xgb.train 
.generate_model <- function(model, data, formula, vars = NULL, add_args = NULL, random_seed = NULL) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)
  
  if (model == "gbm") {
    mod_args <- list()
  } else {
    mod_args <- list(formula = formula, data = data)
  }

  if (model == "logistic") mod_args$family <- "binomial"

  if (!is.null(add_args)) mod_args <- c(mod_args, add_args)
  
  # Prevent default internal scaling for models with the scale parameter
  if (!model %in% c("decisiontree", "gbm", "logistic")) mod_args$scale <- FALSE

  switch(model,
         "lda" = {model <- do.call(lda, mod_args)},
         "qda" = {model <- do.call(qda, mod_args)},
         "logistic" = {model <- do.call(glm, mod_args)},
         "svm" = {model <- do.call(svm, mod_args)},
         "naivebayes" = {model <- do.call(naive_bayes, mod_args)},
         "ann" = {model <- do.call(nnet.formula, mod_args)},
         "knn" = {model <- do.call(train.kknn, mod_args)},
         "decisiontree" = {model <- do.call(rpart, mod_args)},
         "randomforest" = {model <- do.call(randomForest, mod_args)},
         "multinom" = {model <- do.call(multinom, mod_args)},
         "gbm" = {
           mat_data <- data.matrix(data)
           mod_args$data <- xgb.DMatrix(data = mat_data[,vars$predictors], label = mat_data[,vars$target])
           model <- do.call(xgb.train, mod_args)
           }
         )

  return(model)
}

# Helper function for classCV to predict
#' @importFrom stats predict
.prediction <- function(id, mod, train_mod, vars, df_list, thresh, obj, n_classes) {
  # vec to store ground truth and predicted data
  vec <- list("ground" = list(), "pred" = list())
  # Only get predictions for training set if train-test split
  if (id == "split") {
    vec$ground$train <- as.vector(df_list$train[,vars$target])
  } else {
    df_list <- df_list[names(df_list) != "train"]
  }

  vec$ground$test <- as.vector(df_list$test[,vars$target])

  for (i in names(df_list)) {
    switch(mod,
           "lda" = {vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])$class},
           "qda" = {vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])$class},
           "logistic" = {
             vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]], type = "response")
             vec$pred[[i]] <- ifelse(vec$pred[[i]] > thresh, 1, 0)
             },
           "naivebayes" = {vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]][,vars$predictors])},
           "ann" = {vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]], type = "class")},
           "decisiontree" = {
             mat <- predict(train_mod, newdata = df_list[[i]])
             vec$pred[[i]] <- colnames(mat)[apply(mat, 1, which.max)]
             },
           "gbm" = {
             mat <- data.matrix(df_list[[i]])
             xgb_mat <- xgb.DMatrix(data = mat[,vars$predictors],label = mat[,vars$target])
             vec$pred[[i]] <- .handle_gbm_predict(train_mod, xgb_mat, obj, thresh, n_classes)
             },
           # Default for svm, knn, randomforest, and multinom
           vec$pred[[i]] <- predict(train_mod, newdata = df_list[[i]])
    )
    vec$pred[[i]] <- as.vector(vec$pred[[i]])
  }

  return(vec)
}

# Handle different gbm objective functions
.handle_gbm_predict <- function(train_mod, xgb_mat, obj, thresh, n_classes) {
  # produces probability
  bin_prob <- c("reg:logistic", "binary:logistic")
  
  obj <- ifelse(obj %in% bin_prob, "binary_prob", obj)
  
  pred <- predict(train_mod, newdata = xgb_mat)

  # Special cases that need to be converted to labels
  switch(obj,
         "binary_prob" = {pred <- ifelse(pred > thresh, 1, 0)},
         "binary:logitraw" = {
           pred <- sapply(pred, function(x) .logit2prob(x))
           pred <- ifelse(pred > thresh, 1, 0)
         },
         "multi:softprob" = {pred <- max.col(matrix(pred, ncol = n_classes, byrow = TRUE)) - 1}
         )

  return(pred)
}

# Convert logit to probability
.logit2prob <- function(x) {
  prob <- exp(x)/(1 + exp(x))
  return(prob)
}

# Helper function to calculate metrics
.calculate_metrics <- function(class, ground, pred) {
  # Sum of true positives
  true_pos <- sum(ground[which(ground == class)] == pred[which(ground == class)])
  # Sum of false negatives
  false_neg <- sum(ground == class & pred != class)
  # Sum of the false positive
  false_pos <- sum(pred == class) - true_pos
  # Calculate metrics
  precision <- true_pos/(true_pos + false_pos)
  recall <- true_pos/(true_pos + false_neg)
  f1 <- 2*(precision*recall)/(precision + recall)

  metrics <- list("precision" = precision, "recall" = recall,"f1" = f1)
  return(metrics)
}

# Helper function to populate metrics dataframes
.populate_metrics_df <- function(id, classes, vec, met_df) {
  # Create variables used in for loops to calculate precision, recall, and f1

  # Variable to select correct dataframe
  col <- ifelse(id == "split", "Set", "Fold")

  dict <- list("train" = "Training", "test" = "Test")

  for (j in names(vec$pred)) {
    # Calculate classification accuracy
    class_acc <- sum(vec$ground[[j]] == vec$pred[[j]])/length(vec$ground[[j]])
    rowid <- ifelse(col == "Set", dict[[j]], paste(c(col, unlist(strsplit(id, split = "fold"))[2]), collapse = " "))
    met_df[met_df[,col] == rowid,"Classification Accuracy"] <- class_acc
    # class specific metrics
    for (class in classes) {
      met_list <- .calculate_metrics(class, vec$ground[[j]], vec$pred[[j]])
      # Add information to dataframes
      met_df[met_df[,col] == rowid, sprintf("Class: %s Precision", class)] <- met_list$precision
      met_df[met_df[,col] == rowid, sprintf("Class: %s Recall", class)] <- met_list$recall
      met_df[met_df[,col] == rowid, sprintf("Class: %s F-Score", class)] <- met_list$f1
      # Warning is a metric is NA
      met_vals <- c("classification accuracy" = class_acc, "precision" = met_list$precision,
                    "recall" = met_list$recall, "f-score" = met_list$f1)
      if (any(is.na(met_vals))) {
        warning(sprintf("For class %s - %s, the following metrics could not be calculated: '%s'",
                        class, rowid, paste(names(which(is.na(met_vals))), collapse = "', '")))
      }
    }
  }
  if (col == "Fold") met_df <- met_df[met_df[,col] == rowid, colnames(met_df)]

  return(met_df)
}

# Helper function to merge cv dfs
.merge_df <- function(iters, metrics, cv_df) {
  for (i in seq_along(iters)) {
    cv_df[i,colnames(cv_df)] <- metrics[[iters[[i]]]][1,]
  }

  return(cv_df)
}

# Calculate mean, standard deviation, and standard error for cross validation
#' @importFrom stats sd
.get_desc <- function(cv_df, n_folds) {
  indx <- nrow(cv_df)
  desc <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
  cv_df[(indx + 1):(indx + 3),"Fold"] <- desc
  for (colname in colnames(cv_df)[colnames(cv_df) != "Fold"]) {
    # Create vector containing corresponding column name values for each fold
    num_vector <- cv_df[1:indx, colname]
    cv_df[which(cv_df$Fold == desc[1]),colname] <- mean(num_vector, na.rm = TRUE)
    cv_df[which(cv_df$Fold == desc[2]),colname] <- sd(num_vector, na.rm = TRUE)
    cv_df[which(cv_df$Fold == desc[3]),colname] <- sd(num_vector, na.rm = TRUE)/sqrt(n_folds)
  }
  return(cv_df)
}
