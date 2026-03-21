# Calculate string length of classes to create a border of dashed lines
.dashed_lines <- function(classes, return_str = FALSE) {
  str_len <- sapply(classes, nchar)
  max_str_len <- max(str_len)
  partial_output_names <- "Average Precision:  Average Recall:  Average F1:\n\n"
  str <- paste("\nClass:", strrep(" ", max_str_len), partial_output_names)
  cat(rep("-", nchar(str) %/% 1.5), "\n\n\n")

  if (return_str) {
    return(list("max" = max_str_len, "diff" = max_str_len - str_len))
  }
}

# Function to print configs to console
.print_configs <- function(x, model) {
  # Print parameter information
  if (x$configs("n_features") > 20) {
    cat(sprintf("Target: %s\n\n", all.vars(x$configs("formula"))[1]))
  } else {
    str <- capture.output(dput(deparse(x$configs("formula"))))
    str <- gsub("\\s+", " ", paste(str, collapse = ""))
    str <- gsub('\"', "", str)
    cat(sprintf("Formula: %s\n\n", str))
  }

  cat(sprintf("Number of Features: %s\n\n", x$configs("n_features")))
  cat(sprintf("Classes: %s\n\n", paste(x$classes, collapse = ", ")))

  str <- capture.output(dput(x$configs("train_params")))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Training Parameters: %s\n\n", str))

  # Modify model parameters
  info <- x$configs("model_params")

  # Show threshold
  val <- .determine_threshold(
    model, info$map_args$xgboost$params$objective, info$threshold, FALSE
  )
  if (!is.null(val)) info$threshold <- val

  if (!startsWith(model, "regularized") ||
    (startsWith(model, "regularized") && is.null(info$rule))) {
    info <- info[!names(info) %in% c("rule", "verbose")]
  }

  info$map_args <- info$map_args[!names(info$map_args) != model]

  if (length(info$map_args) == 0) {
    info$map_args <- NULL
    info <- c(list(map_args = NULL), info)
  }

  str <- capture.output(dput(info))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Model Parameters: %s\n\n", str))

  # Print sample size and missing data for user transparency
  missing_data_summary <- x$get_missing_data_summary()
  complete_obs <- missing_data_summary$complete_observations
  incomplete_obs <- missing_data_summary$incomplete_labeled_observations

  cat(sprintf(
    "Unlabeled Observations: %s\n\n",
    missing_data_summary$unlabeled_observations
  ))
  cat(sprintf("Incomplete Labeled Observations: %s\n\n", incomplete_obs))
  cat(sprintf(
    "Observations Missing All Features: %s\n\n",
    missing_data_summary$observations_missing_all_features
  ))

  if (!is.null(x$configs("impute_params", "method"))) {
    total <- complete_obs + incomplete_obs
    cat(sprintf(
      "Sample Size (Complete + Imputed Incomplete Labeled Observations): %s\n\n",
      total
    ))
  } else {
    cat(sprintf("Sample Size (Complete Observations): %s\n\n", complete_obs))
  }

  str <- capture.output(dput(x$configs("impute_params")))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Imputation Parameters: %s\n\n", str))

  str <- capture.output(dput(x$configs("parallel_configs")))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Parallel Configs: %s\n\n", str))
}

.format_metric <- function(value) {
  if (is.na(value)) {
    return("NaN")
  }
  format(round(value, 2), nsmall = 2)
}


.format_mean_sd <- function(mean_val, sd_val) {
  sprintf(
    "%s \u00B1 %s (SD)",
    .format_metric(mean_val),
    .format_metric(sd_val)
  )
}

# Helper to get class-specific column names from a metrics dataframe
.get_class_columns <- function(class, data) {
  class_cols <- c()
  for (colname in colnames(data)) {
    split_colname <- unlist(strsplit(colname, split = " "))
    class_name_components <- split_colname[-c(1, length(split_colname))]
    split_classname <- unlist(strsplit(class, split = " "))
    if (all(split_classname %in% class_name_components) &&
      length(setdiff(class_name_components, split_classname)) == 0) {
      class_cols <- c(class_cols, colname)
    }
  }

  return(class_cols)
}

# Function to print train-test split metrics to console
.print_metrics_split <- function(x, model, max_str_len) {
  data <- x$metrics(model, "split")

  met_width <- 14

  for (set in data$Set) {
    cat(sprintf("\n\n %s\n", set))
    cat(rep("_", 21), "\n\n")

    acc <- data[data$Set == set, "Classification Accuracy"]
    cat(sprintf("  Classification Accuracy: %s\n\n", .format_metric(acc)))

    # sprintf has alignment patterns %-*s, is dynamic left alignment
    cat(sprintf(
      "  %-*s  %-*s  %-*s  %s\n\n",
      max_str_len + 6, "Class:",
      met_width, "Precision:",
      met_width, "Recall:",
      "F1:"
    ))

    for (class in x$classes) {
      class_cols <- .get_class_columns(class, data)
      vals <- sapply(data[data$Set == set, class_cols], .format_metric)

      cat(sprintf(
        "  %-*s  %-*s  %-*s  %s\n",
        max_str_len + 6, class,
        met_width, vals[1],
        met_width, vals[2],
        vals[3]
      ))
    }
  }
}

# Function to print cross validation metrics to console
.print_metrics_cv <- function(x, model, max_str_len) {
  data <- x$metrics(model, "cv")

  cat(sprintf("\n\n Cross-validation (CV)\n"))
  cat(rep("_", 21), "\n\n")

  # Classification accuracy
  mean_acc <- data[data$Fold == "Mean CV:", "Classification Accuracy"]
  sd_acc <- data[data$Fold == "Standard Deviation CV:", "Classification Accuracy"]
  cat(sprintf(
    "  Average Classification Accuracy: %s\n\n",
    .format_mean_sd(mean_acc, sd_acc)
  ))

  met_width <- 24

  cat(sprintf(
    "  %-*s  %-*s  %-*s  %s\n\n",
    max_str_len + 6, "Class:",
    met_width, "Average Precision:",
    met_width, "Average Recall:",
    "Average F1:"
  ))

  for (class in x$classes) {
    class_cols <- .get_class_columns(class, data)

    mean_vals <- as.numeric(data[x$n_folds + 1, class_cols])
    sd_vals <- as.numeric(data[x$n_folds + 2, class_cols])

    formatted <- sapply(seq_along(mean_vals), function(i) {
      .format_mean_sd(mean_vals[i], sd_vals[i])
    })

    cat(sprintf(
      "  %-*s  %-*s  %-*s  %s\n",
      max_str_len + 6, class,
      met_width, formatted[1],
      met_width, formatted[2],
      formatted[3]
    ))
  }
}
