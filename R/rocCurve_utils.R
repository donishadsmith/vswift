# Helper function to perform checks to ensure information needed is available and to obtain information needed for plotting
.perform_checks <- function(x, data) {
  if (is.null(x$models)) {
    stop("models must be saved in order to use `rocCurve`")
  }

  # Check if data is available
  if (!is.data.frame(data) && is.null(x$data_partitions$dataframes)) {
    stop("data cannot be NULL if dataframes were not saved by `classCV`")
  }

  # Check if target is binary
  df <- .get_data(x, data)$data

  vars <- .get_var_names(formula = x$configs$formula, data = df)

  if (length(x$class_summary$classes) != 2) {
    stop("`rocCurve` currently only supports binary targets")
  }

  # Convert target
  class_keys <- .create_dictionary(x$class_summary$classes, TRUE)

  return(list("keys" = class_keys, "vars" = vars))
}

# Helper function to get data, indices, and models
.get_data <- function(x, data, id = NULL, foldid = NULL, get_indices = FALSE, vars = NULL, model = NULL,
                      discard_labels = TRUE) {
  preprocess <- ifelse(is.data.frame(data), TRUE, FALSE)
  # Get information for indexing for either dataframes or the test set indices
  id <- ifelse(is.null(id), names(x$data_partitions$indices)[1], id)

  if (!is.null(x$data_partitions$indices$cv)) {
    foldid <- ifelse(is.null(foldid), names(x$data_partitions$indices$cv)[1], foldid)
  }

  # Get data
  if (is.data.frame(data)) {
    df <- data
    rownames(df) <- seq(nrow(df))
    # Discard missing labels
    if (discard_labels) {
      target <- which(is.na(df[, all.vars(x$configs$formula)[1]]))
      df <- df[-unique(target), ]
    }
  } else {
    if (id == "split") {
      df <- rbind(
        x$data_partitions$dataframes$split$train, x$data_partitions$dataframes$split$test
      )
    } else {
      df <- rbind(
        x$data_partitions$dataframes$cv[[foldid]]$train, x$data_partitions$dataframes$cv[[foldid]]$test
      )
    }
  }

  # Sort rows if data extracted from vswift object
  if (!is.data.frame(data)) df <- df[order(as.numeric(rownames(df))), ]

  # Ensure all characters are factors
  if (isTRUE(preprocess) && !is.null(vars)) {
    out <- .convert_to_factor(df, vars$target, model, remove_obs = FALSE)
    missing_info <- .missing_summary(out$data, vars$target)
    impute <- ifelse(!is.null(x$imputation_models), TRUE, FALSE)
    cleaned_data <- .clean_data(df, missing_info, impute, FALSE)
    out$data <- cleaned_data$cleaned_data
  } else {
    out <- list("data" = df)
  }

  # Get the test set
  if (get_indices) {
    indices <- if (id == "split") x$data_partitions$indices$split$test else x$data_partitions$indices$cv[[foldid]]
    out$indices <- indices
  }

  return(out)
}

.quick_prep <- function(x, df_list, id, foldid, info, preprocess) {
  # Check imputation first
  if (!is.null(x$imputation_models) && isTRUE(preprocess)) {
    prep <- if (id == "split") x$imputation_models$split else x$imputation_models$cv[[foldid]]
    df_list <- .impute_bake(train = df_list$train, test = df_list$test, vars = info$vars, prep = prep)
  }

  # Determine if standardizing is needed
  standardize <- ((isTRUE(x$configs$train_params$standardize) || is.numeric(x$configs$train_params$standardize)) &&
    is.null(x$imputation_models))
  # Check if standardized need standardized
  if (standardize) {
    df_list <- .standardize_train(
      df_list$train, df_list$test,
      standardize = x$configs$train_params$standardize, info$vars$target
    )
  }

  return(df_list)
}

.computeROC <- function(x, data, model, plot_title, split, cv, thresholds, info, path, ...) {
  out <- list()

  if (isTRUE(split) && !is.null(x$configs$train_params$split)) {
    out$split <- .get_thresholds(x, data, "split", NULL, model, thresholds, info)

    for (i in c("train", "test")) {
      # Get fpr and tpr
      out$split[[i]]$metrics <- .compute_scores(
        out$split[[i]]$prob, out$split[[i]]$thresholds, out$split[[i]]$labels
      )
      # Perform integration with trapezoidal rule to obtain auc
      out$split[[i]]$auc <- .integrate(out$split[[i]]$metrics$fpr, out$split[[i]]$metrics$tpr)
      # Obtain Youdin's Index
      out$split[[i]]$youdins_indx <- .youdins_indx(
        out$split[[i]]$metrics$fpr, out$split[[i]]$metrics$tpr, out$split[[i]]$thresholds
      )
    }

    # Plot curves
    .plot_curve(out$split, "train_test", model, path, ...)
  }

  if (isTRUE(cv) && !is.null(x$configs$train_params$n_folds)) {
    for (foldid in paste0("fold", seq(x$configs$train_params$n_folds))) {
      out$cv[[foldid]] <- .get_thresholds(x, data, "cv", foldid, model, thresholds, info)

      # Get fpr and tpr
      out$cv[[foldid]]$metrics <- .compute_scores(
        out$cv[[foldid]]$prob, out$cv[[foldid]]$thresholds, out$cv[[foldid]]$labels
      )
      # Perform integration with trapezoidal rule to obtain auc
      out$cv[[foldid]]$auc <- .integrate(out$cv[[foldid]]$metrics$fpr, out$cv[[foldid]]$metrics$tpr)
      # Obtain Youdin's Index
      out$cv[[foldid]]$youdins_indx <- .youdins_indx(
        out$cv[[foldid]]$metrics$fpr, out$cv[[foldid]]$metrics$tpr, out$cv[[foldid]]$thresholds
      )
    }

    # Plot curves
    .plot_curve(out$cv, "cv", model, path, ...)
  }

  return(out)
}

# Helper function to obtain thresholds used for ROC curve
.get_thresholds <- function(x, data, id, foldid = NULL, model, thresholds, info) {
  preprocess <- ifelse(is.data.frame(data), TRUE, FALSE)
  # Get training model
  if (id == "split") {
    train_mod <- x$models[[model]]$split
  } else {
    train_mod <- x$models[[model]]$cv[[foldid]]
  }

  # Get data
  out <- .get_data(x, data, id, foldid, TRUE, info$vars, model, FALSE)

  # Partition training and test data
  df_list <- .partition(out$data, out$indices)

  if (preprocess) df_list <- .quick_prep(x, df_list, id, foldid, info, preprocess)

  # Ensure factored columns have same levels for svm
  if (model == "svm" && !is.null(out$col_levels)) {
    col_levels <- out$col_levels
    for (i in names(df_list)) {
      df_list[[i]][, names(col_levels)] <- data.frame(
        lapply(names(col_levels), function(col) factor(df_list[[i]][, col], levels = col_levels[[col]]))
      )
    }
  }

  pred_list <- .prediction(
    id, model, train_mod, info$vars, df_list, NULL, x$configs$model_params$map_args$xgboost$params$objective,
    length(info$keys),
    keep_probs = TRUE
  )

  # Handle prediction output, some models will produce a matrix with posterior probabilities for both outcomes
  for (name in names(pred_list$pred)) {
    if (!(model %in% c("logistic", "regularized_logistic", "nnet", "multinom", "xgboost"))) {
      # Return the column corresponding to one in keys
      if (model != "xgboost") {
        pred_list$pred[[name]] <- pred_list$pred[[name]][, names(info$keys)[info$keys == 1]]
      } else {
        pred_list$pred[[name]] <- pred_list$pred[[name]][, 1]
      }
    }
    pred_list$pred[[name]] <- as.vector(pred_list$pred[[name]])
  }

  # Out list
  out <- list()

  for (name in names(pred_list$pred)) {
    if (!is.null(thresholds)) {
      out[[name]]$thresholds <- unique(sort(c(0, thresholds)))
    } else {
      probs <- pred_list$pred[[name]]
      probs <- c(0, probs)
      out[[name]]$thresholds <- unique(sort(probs))
    }
    out[[name]]$probs <- pred_list$pred[[name]]

    if (inherits(pred_list$ground[[name]], "character")) {
      out[[name]]$labels <- unlist(Map(function(x) info$key[[x]], pred_list$ground[[name]]))
    }
  }

  # For ids that start with fold, unnest
  if (id == "cv") {
    out <- list("thresholds" = out$test$thresholds, "probs" = out$test$probs, "labels" = out$test$labels)
  }

  return(out)
}

# Helper function to compute true and false positive rates
.compute_scores <- function(probs, thresholds, ground) {
  # Create outer product matrix; rows = probs and cols = thresholds
  mat <- outer(probs, thresholds, ">=")
  # Sum of all columns to get the true positives for all thresholds
  true_pos <- colSums(mat[ground == 1, ])
  # Subtract column sums from true_pos to obtain false_pos
  false_pos <- colSums(mat) - true_pos
  # Compute tpr
  tpr <- true_pos / sum(ground)
  # Compute fpr
  fpr <- false_pos / sum(!ground)

  return(list("fpr" = fpr, "tpr" = tpr))
}

# Helper function to compute area under the curve
.integrate <- function(fpr, tpr) {
  paired_list <- Map(list, "fpr" = fpr, "tpr" = tpr)
  # Order by decreasing to ascending for fpr and the reverse of tpr; Each first instance of duplicate fpr is the max tpr
  paired_list_ordered <- paired_list[order(
    sapply(paired_list, function(x) x$fpr),
    -sapply(paired_list, function(x) x$tpr)
  )]
  # Obtain the fpr values, determine which is not duplicated to retain only the first instance
  paired_list_final <- paired_list_ordered[!duplicated(sapply(paired_list_ordered, function(x) x$fpr))]
  N <- length(paired_list_final) - 1
  # sum all areas to compute total area = auc
  auc <- sum(sapply(seq(N), function(x) .trapezoid(paired_list_final[[x]], paired_list_final[[x + 1]])))

  return(auc)
}

# Helper function to perform the trapezoidal rule
.trapezoid <- function(p1, p2) {
  # width and height of trapezoid
  area <- (p2$fpr - p1$fpr) * (p2$tpr + p1$tpr) / 2

  return(area)
}

# Helper function to implement youdin's index
.youdins_indx <- function(fpr, tpr, thresholds) {
  # j = sensitivity + specificity - 1
  # tpr = sensitivity; fpr = 1 - specificity
  # sensitivity - (1 - specificity) = sensitivity + specificity - 1 = tpr - fpr
  j <- tpr - fpr
  max_diff <- max(j)
  optimal_threshold <- thresholds[j == max_diff]

  return(optimal_threshold)
}

# Helper function to plot curves
.plot_curve <- function(object, method, model, path, ...) {
  # Save if path detected
  if (!is.null(path)) {
    # Get OS separator
    os.sep <- ifelse(as.vector(Sys.info()["sysname"] == "Windows"), "\\", "/")
    full_png_name <- sprintf("%s_%s_roc_curve.png", model, method)
    filename <- paste0(path, os.sep, full_png_name)
    png(filename = filename, ...)
  }

  plot(
    NULL,
    xlim = c(0, 1), ylim = c(0, 1), xaxp = c(0, 1, 10), yaxp = c(0, 1, 10),
    xlab = "False Positive Rate (FPR)", ylab = "True Positive Rate (TPR)",
    main = sprintf("%s - ROC Curve", .MODEL_LIST[[model]])
  )

  colors <- rainbow(length(names(object)), alpha = 1)

  legend_labels <- c()
  legend_colors <- c()
  legend_lty <- c()

  for (i in seq_along(names(object))) {
    # Get proper name
    id <- names(object)[i]
    proper_id <- .get_proper_name(id)

    # Append to legend vectors
    auc <- object[[id]]$auc
    legend_labels <- c(legend_labels, sprintf("%s (AUC: %s)", proper_id, round(auc, 2)))
    legend_colors <- c(legend_colors, colors[i])
    legend_lty <- c(legend_lty, 1)

    # Get lines
    lines(object[[id]]$metrics$fpr, object[[id]]$metrics$tpr, col = colors[i], lty = 1)
  }

  # Append to legend vectors
  legend_labels <- c(legend_labels, "Random Classifier")
  legend_colors <- c(legend_colors, "black")
  legend_lty <- c(legend_lty, 2)

  legend("bottomright", legend = legend_labels, col = legend_colors, lty = legend_lty)

  # Dashed line
  abline(a = 0, b = 1, lty = 2, col = "black")

  # Use dev.new for certain R environments or dev.off if png is used
  .display(path)
}

# Helper function to obtain a proper name
.get_proper_name <- function(id) {
  names_list <- list("train" = "Training", "test" = "Test")

  if (startsWith(id, "fold")) {
    proper_name <- paste("Fold", unlist(strsplit(id, split = "fold"))[2])
  } else {
    proper_name <- names_list[[id]]
  }

  return(proper_name)
}
