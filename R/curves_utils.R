# Helper function that servess as the entry point for roc and pr curves
.curve_entry <- function(x,
                         data = NULL,
                         models = NULL,
                         split = TRUE,
                         cv = TRUE,
                         thresholds = NULL,
                         return_output = TRUE,
                         curve_method,
                         path = NULL, ...) {
  if (inherits(x, "vswift")) {
    # Perform checks and get dictionary class keys and variables
    info <- .perform_checks(x, data, curve_method)

    # Unlist keys to turn into a named vector
    info$keys <- unlist(info$keys)

    # Get valid models
    models <- .intersect_models(x, models)

    if ("xgboost" %in% models && x$configs$model_params$map_args$xgboost$params$objective == "multi:softmax") {
      warnings("'xgboost' cannot be specified when the 'multi:softmax; objective is used since probabilties are needed")
      models <- models[!models == "xgboost"]
    }

    if ("xgboost" %in% models && x$configs$model_params$map_args$xgboost$params$objective == "binary:hinge") {
      if (is.null(thresholds)) stop("`thresholds` must be specified since 'xgboost' uses the 'binary:hinge' objective")
    }

    if (length(models) == 0) stop("no valid models to plot")

    # Iterate over models
    output <- list()

    for (model in models) {
      output[[model]] <- .curve_pipeline(
        x, data, model, .MODEL_LIST[[model]], split, cv, thresholds, info, curve_method, path, ...
      )

      if (!isTRUE(return_output)) output[[model]] <- NULL
    }

    if (isTRUE(return_output)) {
      return(output)
    }
  } else {
    stop("`x` must be an object of class 'vswift'")
  }
}

# Helper function to perform checks to ensure information needed is available and to obtain information needed for plotting
.perform_checks <- function(x, data, curve_method) {
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
  class_keys <- .create_dictionary(x$class_summary$classes, alternate_warning = TRUE, curve_method = curve_method)

  return(list("keys" = class_keys, "vars" = vars))
}

# Helper function to get data, indices, and models
.get_data <- function(x, data, id = NULL, foldid = NULL, get_indices = FALSE, vars = NULL, model = NULL,
                      discard_unusable_data = TRUE) {
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
    if (discard_unusable_data) {
      miss_info <- .missing_summary(data, all.vars(x$configs$formula)[1])
      discard_indices <- c(miss_info$unlabeled_data_indices, miss_info$missing_all_features_indices)
      if (length(discard_indices) != 0) df <- df[-discard_indices, ]
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
    miss_info <- .missing_summary(out$data, vars$target)
    impute <- ifelse(!is.null(x$imputation_models), TRUE, FALSE)
    cleaned_data <- .clean_data(out$data, miss_info, impute, FALSE)
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

# Helper function to perform quick preparation of input dataframe
.quick_prep <- function(x, df_list, id, foldid, info, preprocess, model, col_levels) {
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

  # Relevel columns if svm
  if (model == "svm" && !is.null(col_levels)) {
    for (i in names(df_list)) df_list[[i]] <- .relevel_cols(df_list[[i]], col_levels)
  }

  return(df_list)
}

# Helper function that serves as the pipeline for producing curves
.curve_pipeline <- function(x, data, model, plot_title, split, cv, thresholds, info, curve_method, path, ...) {
  out <- list()

  if (isTRUE(split) && !is.null(x$configs$train_params$split)) {
    out$split <- .get_thresholds(x, data, "split", NULL, model, thresholds, info)

    for (i in c("train", "test")) {
      out$split[[i]] <- c(out$split[[i]], .get_curve_metrics(out$split[[i]], curve_method))
      # Rename tpr to recall
      if (curve_method != "roc") names(out$split[[i]]$metrics) <- .rename_metrics(out$split[[i]]$metrics)
    }

    # Plot curves
    .plot_curve(out$split, curve_method, "train_test", model, path, ...)
  }

  if (isTRUE(cv) && !is.null(x$configs$train_params$n_folds)) {
    for (foldid in paste0("fold", seq(x$configs$train_params$n_folds))) {
      out$cv[[foldid]] <- .get_thresholds(x, data, "cv", foldid, model, thresholds, info)
      out$cv[[foldid]] <- c(out$cv[[foldid]], .get_curve_metrics(out$cv[[foldid]], curve_method))
      # Rename tpr to recall
      if (curve_method != "roc") names(out$cv[[foldid]]$metrics) <- .rename_metrics(out$cv[[foldid]]$metrics)
    }

    # Plot curves
    .plot_curve(out$cv, curve_method, "cv", model, path, ...)
  }

  return(out)
}

# Helper function to obtain curve specific information
.get_curve_metrics <- function(x, curve_method) {
  out <- list()
  # Get metrics
  out$metrics <- .compute_scores(x$prob, x$thresholds, x$labels, curve_method)

  # Obtain AUC and Youden's Index or Max F1
  if (curve_method == "roc") {
    out$auc <- .integrate(fpr = out$metrics$fpr, tpr = out$metrics$tpr, curve_method = curve_method)
    out$youdens_indx <- .youdens_indx(out$metrics$fpr, out$metrics$tpr, x$thresholds)
  } else {
    out$auc <- .integrate(precision = out$metrics$precision, tpr = out$metrics$tpr, curve_method = curve_method)
    scores <- .maxf1(out$metrics$tpr, out$metrics$precision, x$thresholds)
    out$maxF1 <- scores$maxF1
    out$optimal_threshold <- scores$optimal_threshold
  }

  return(out)
}

# Helper function to rename tpr -> recall
.rename_metrics <- function(metrics) {
  curr_names <- names(metrics)
  curr_names[curr_names == "tpr"] <- "recall"

  return(curr_names)
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

  if (preprocess) df_list <- .quick_prep(x, df_list, id, foldid, info, preprocess, model, out$col_levels)

  results <- .prediction(
    id, model, train_mod, info$vars, df_list, NULL, x$configs$model_params$map_args$xgboost$params$objective,
    length(info$keys),
    probs = TRUE, keys = info$keys, caller = "curve"
  )

  # Out list
  out <- list()

  for (name in names(results$pred)) {
    out[[name]]$thresholds <- if (!is.null(thresholds)) thresholds else results$pred[[name]]
    out[[name]]$thresholds <- c(0, out[[name]]$thresholds)
    out[[name]]$thresholds <- unique(sort(out[[name]]$thresholds))
    out[[name]]$probs <- results$pred[[name]]

    if (inherits(results$ground[[name]], "character")) {
      out[[name]]$labels <- unlist(Map(function(x) info$key[[x]], results$ground[[name]]))
    }
  }

  # For ids that start with fold, unnest
  if (id == "cv") {
    out <- list("thresholds" = out$test$thresholds, "probs" = out$test$probs, "labels" = out$test$labels)
  }

  return(out)
}

# Helper function to compute true positive rates (recall) and false positive rates for each
# threshold using outer product matrix
.compute_scores <- function(probs, thresholds, ground, curve_method) {
  # Create outer product matrix; rows = probs and cols = thresholds
  mat <- outer(probs, thresholds, ">=")
  # Sum of all columns to get the true positives for all thresholds
  true_pos <- colSums(mat[ground == 1, ])
  # Subtract column sums from true_pos to obtain false_pos
  false_pos <- colSums(mat) - true_pos
  # Compute tpr
  tpr <- true_pos / sum(ground)

  if (curve_method == "roc") {
    # Compute fpr
    fpr <- false_pos / sum(!ground)
    return(list("fpr" = fpr, "tpr" = tpr))
  } else {
    precision <- (true_pos) / (true_pos + false_pos)
    # Handle zero division
    precision[is.na(precision)] <- 0
    return(list("tpr" = tpr, "precision" = precision))
  }
}

# Helper function to turn a paired_list to a simple list with two levels
.simplify_list <- function(paired_list) {
  x <- sapply(paired_list, function(coord) coord$x)
  y <- sapply(paired_list, function(coord) coord$y)

  return(list("x" = x, "y" = y))
}

# Helper function to add anchor for prCurve
.add_anchor <- function(x, y, plot = FALSE) {
  paired_list <- Map(list, "x" = x, "y" = y)
  # Sort
  paired_list <- paired_list[order(sapply(paired_list, function(coord) coord$x))]
  coords <- .simplify_list(paired_list)

  if (coords$y[1] == 0) {
    coords$x <- coords$x[-1]
    coords$y <- coords$y[-1]
  }

  # Add zero to x
  if (coords$x[1] != 0) {
    coords$x <- c(0, coords$x)
    coords$y <- c(1, coords$y)
  }

  if (plot) {
    paired_list <- Map(list, "x" = coords$x, "y" = coords$y)
    paired_list_ordered <- .order_paired_list(paired_list, c("x", "y"))
    coords <- .simplify_list(paired_list_ordered)
  }

  return(list("x" = coords$x, "y" = coords$y))
}

# Helper function to order a paired list
.order_paired_list <- function(paired_list, order_names) {
  paired_list_ordered <- paired_list[order(
    sapply(paired_list, function(x) x[[order_names[1]]]),
    -sapply(paired_list, function(x) x[[order_names[2]]])
  )]

  return(paired_list_ordered)
}

# Helper function to order create and order a paired list
.create_paired_list <- function(fpr = NULL, precision = NULL, tpr, curve_method) {
  # Add 0, 1 point to cover full integration
  if (curve_method == "roc") {
    paired_list <- Map(list, "fpr" = fpr, "tpr" = tpr)
    order_names <- c("fpr", "tpr")
  } else {
    # Add anchor temporarily for integration
    points <- .add_anchor(tpr, precision)
    paired_list <- Map(list, "tpr" = points$x, "precision" = points$y)
    order_names <- c("tpr", "precision")
  }

  # Order by decreasing -> increasing for metric that is x-axis (fpr for roc and tpr for pr)
  # For metric that is y-axis order from increasing -> decreasing
  # Each first instance of duplicated pairs will have the minimum x paired with the maximum y
  paired_list_ordered <- .order_paired_list(paired_list, order_names)

  i <- ifelse(curve_method == "roc", "fpr", "tpr")
  # Obtain the fpr or tpr values, determine which is not duplicated to retain only the first instance
  paired_list_final <- paired_list_ordered[!duplicated(sapply(paired_list_ordered, function(x) x[[i]]))]

  return(paired_list_final)
}

# Helper function to compute area under the curve
.integrate <- function(fpr = NULL, precision = NULL, tpr, curve_method) {
  paired_list <- .create_paired_list(fpr, precision, tpr, curve_method)
  N <- length(paired_list) - 1
  # sum all areas to compute total area = auc
  auc <- sum(sapply(seq(N), function(x) .trapezoid(paired_list[[x]], paired_list[[x + 1]], curve_method)))

  return(auc)
}

# Helper function to perform the trapezoidal rule
.trapezoid <- function(p1, p2, curve_method) {
  # Width and height of trapezoid
  if (curve_method == "roc") {
    area <- (p2$fpr - p1$fpr) * (p2$tpr + p1$tpr) / 2
  } else {
    area <- (p2$tpr - p1$tpr) * (p2$precision + p1$precision) / 2
  }

  return(area)
}

# Helper function to implement youden's index
.youdens_indx <- function(fpr, tpr, thresholds) {
  # j = sensitivity + specificity - 1
  # tpr = sensitivity; fpr = 1 - specificity
  # sensitivity - (1 - specificity) = sensitivity + specificity - 1 = tpr - fpr
  j <- tpr - fpr
  max_diff <- max(j)
  optimal_threshold <- thresholds[j == max_diff]

  return(optimal_threshold)
}

# Helper function to compute threshold with max index
.maxf1 <- function(recall, precision, thresholds) {
  # Compute F1 scores
  f1_scores <- unlist(Map(function(x, y) (2 * x * y) / (x + y), precision, recall))
  # Select index of max F1 score
  max_indx <- which.max(f1_scores)

  return(list("maxF1" = f1_scores[max_indx], "optimal_threshold" = thresholds[max_indx]))
}

# Helper function to plot curves
.plot_curve <- function(object, curve_method, method, model, path, ...) {
  names <- .curve_names(curve_method)

  # Save if path detected
  if (!is.null(path)) {
    # Get OS separator
    os.sep <- ifelse(as.vector(Sys.info()["sysname"] == "Windows"), "\\", "/")
    full_png_name <- sprintf("%s_%s_%s_curve.png", model, method, names$png)
    filename <- paste0(path, os.sep, full_png_name)
    png(filename = filename, ...)
  }

  plot(
    NULL,
    xlim = c(0, 1), ylim = c(0, 1), xaxp = c(0, 1, 10), yaxp = c(0, 1, 10),
    xlab = sprintf("%s", names$x), ylab = sprintf("%s", names$y),
    main = sprintf("%s - %s Curve", .MODEL_LIST[[model]], names$main)
  )

  colors <- rainbow(length(names(object)), alpha = 1)

  legend_labels <- c()
  legend_colors <- c()
  legend_lty <- c()
  prop_pos <- c()

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
    if (curve_method == "roc") {
      x <- "fpr"
      y <- "tpr"
    } else {
      x <- "recall"
      y <- "precision"
    }

    coords <- list("x" = object[[id]]$metrics[[x]], "y" = object[[id]]$metrics[[y]])

    # Add anchor temporarily for plotting
    if (curve_method == "pr") {
      coords <- .add_anchor(coords$x, coords$y, TRUE)
      # Append the proportion of positive instances
      prop_pos <- c(prop_pos, sum(object[[id]]$labels == 1) / length(object[[id]]$labels))
    }

    lines(coords$x, coords$y, col = colors[i], lty = 1)
  }

  # Append to legend vectors
  legend_labels <- c(legend_labels, "Random Classifier")
  legend_colors <- c(legend_colors, "black")
  legend_lty <- c(legend_lty, 2)

  legend("bottomright", legend = legend_labels, col = legend_colors, lty = legend_lty)

  # Dashed line
  if (curve_method == "roc") {
    abline(a = 0, b = 1, lty = 2, col = "black")
  } else {
    abline(h = mean(prop_pos), lty = 2, col = "black")
  }

  # Use dev.new for certain R environments or dev.off if png is used
  .display(path)
}

# Helper information to obtain axes, png, and main name
.curve_names <- function(curve_method) {
  if (curve_method == "roc") {
    names <- list(
      "main" = "ROC", "png" = "roc", "x" = "False Positive Rate (FPR)", "y" = "True Positive Rate (TPR)"
    )
  } else {
    names <- list(
      "main" = "Precision-Recall", "png" = "precision_recall", "x" = "Recall", "y" = "Precision"
    )
  }

  return(names)
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
