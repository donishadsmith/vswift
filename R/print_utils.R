# Function to print configs to console
.print_configs <- function(x, model) {
  # Print parameter information
  if (x$configs$n_features > 20) {
    cat(sprintf("Target: %s\n\n", all.vars(x$configs$formula)[1]))
  } else {
    str <- capture.output(dput(deparse(x$configs$formula)))
    str <- gsub("\\s+", " ", paste(str, collapse = ""))
    str <- gsub('\"', "", str)
    cat(sprintf("Formula: %s\n\n", str))
  }
  cat(sprintf("Number of Features: %s\n\n", x$configs$n_features))
  cat(sprintf("Classes: %s\n\n", paste(x$class_summary$classes, collapse = ", ")))
  str <- capture.output(dput(x$configs$train_params))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Training Parameters: %s\n\n", str))

  # Modify model parameters
  info <- x$configs$model_params

  # Show threshold
  val <- .determine_threshold(model, info$map_args$xgboost$params$objective, info$threshold, FALSE)
  if (!is.null(val)) info$threshold <- val

  if (!startsWith(model, "regularized") || (startsWith(model, "regularized") && is.null(info$rule))) {
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
  cat(sprintf("Unlabeled Observations: %s\n\n", x$missing_data_summary$unlabeled_observations))
  cat(sprintf("Incomplete Labeled Observations: %s\n\n", x$missing_data_summary$incomplete_labeled_observations))
  cat(sprintf("Observations Missing All Features: %s\n\n", x$missing_data_summary$observations_missing_all_features))
  if (!is.null(x$configs$impute_params$method)) {
    total <- x$missing_data_summary$complete_observations + x$missing_data_summary$incomplete_labeled_observations
    cat(sprintf("Sample Size (Complete + Imputed Incomplete Labeled Observations): %s\n\n", total))
  } else {
    cat(sprintf("Sample Size (Complete Observations): %s\n\n", x$missing_data_summary$complete_observations))
  }
  str <- capture.output(dput(x$configs$impute_params))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Imputation Parameters: %s\n\n", str))

  # Print information for parallel processing
  str <- capture.output(dput(x$configs$parallel_configs))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Parallel Configs: %s\n\n", str))
}

# Function to print train-test split metrics to console
.print_metrics_split <- function(x, df, max_str_len, str_diff) {
  for (set in c("Training", "Test")) {
    # Variable for which class string length to print to ensure all values have equal spacing
    class_pos <- 1
    # Print name of the set metrics to be printed and add underscores
    cat("\n\n", set, "\n")
    cat(rep("_", 21), "\n\n")
    # Print classification accuracy
    cat("Classification Accuracy: ", format(round(df[df$Set == set, "Classification Accuracy"], 2), nsmall = 2), "\n\n")
    # Print name of metrics
    cat("Class:", rep("", max_str_len - 1), "Precision:", "", "Recall:", strrep(" ", 5), "F1:\n\n")
    # For loop to obtain vector of values for each class

    for (class in x$class_summary$classes) {
      # Get class specific columns
      class_cols <- .split_colnames(class, df)

      # Print metric corresponding to class
      class_met <- sapply(df[df$Set == set, class_cols], function(x) format(round(x, 2), nsmall = 2))
      # Add spacing
      padding <- nchar(paste("Class:", "", "Pre"))

      # Pad output with strings
      formatted_class_met <- c()

      for (i in seq_along(class_met)) {
        formatted_class_met <- c(formatted_class_met, class_met[i])
        if (i != length(class_met)) {
          if (i == 1) space <- if (class_met[i] != "NaN") rep("", 4) else rep("", 5)
          if (i == 2) space <- if (class_met[i] != "NaN") rep("", 5) else rep("", 6)
          formatted_class_met <- c(formatted_class_met, space)
        }
      }

      cat(class, rep("", (padding + str_diff[class_pos])), paste(formatted_class_met, collapse = " "), "\n")

      class_pos <- class_pos + 1
    }
  }
}

# Function to print cross validation metrics to console
.print_metrics_cv <- function(x, df, max_str_len, str_diff) {
  # Variable for which class string length to print to ensure all values have equal spacing
  class_pos <- 1
  # Get number of folds to select the correct rows for mean and stdev
  n_folds <- x$configs$train_params$n_folds
  # Print parameters name
  cat("\n\n", "Cross-validation (CV)", "\n")
  cat(rep("_", 21), "\n\n")
  mean_cv <- round(df[df$Fold == "Mean CV:", "Classification Accuracy"], 2)
  sd_cv <- round(df[df$Fold == "Standard Deviation CV:", "Classification Accuracy"], 2)
  acc_met <- c(format(mean_cv, nsmall = 2), format(sd_cv, nsmall = 2))

  acc_met <- sprintf("%s \U00B1 %s (SD)", acc_met[1], acc_met[2])
  cat("Average Classification Accuracy: ", acc_met, "\n\n")
  cat(
    "Class:", rep("", max_str_len), strrep(" ", 2), "Average Precision:", strrep(" ", 6),
    "Average Recall:", strrep(" ", 10), "Average F1:\n\n"
  )

  # Go through column names, split the colnames and class name to see if the column name is the metric for that class
  for (class in x$class_summary$classes) {
    # Get class specific columns
    class_cols <- .split_colnames(class, df)

    # Print metric corresponding to class
    mean_met <- sapply(df[((n_folds + 1)), class_cols], function(x) format(round(x, 2), nsmall = 2))
    sd_met <- sapply(df[((n_folds + 2)), class_cols], function(x) format(round(x, 2), nsmall = 2))
    sd_met_pos <- 1
    class_met <- c()

    for (metric in mean_met) {
      class_met <- c(class_met, sprintf("%s \u00B1 %s (SD)", metric, sd_met[sd_met_pos]))
      sd_met_pos <- sd_met_pos + 1
    }

    # Pad output with strings
    formatted_class_met <- c()
    for (i in seq_along(class_met)) {
      formatted_class_met <- c(formatted_class_met, class_met[i])
      space <- if (class_met[i] == "NaN \u00B1 NA (SD)") rep("", 9) else rep("", 6)
      if (i != length(class_met)) formatted_class_met <- c(formatted_class_met, space)
    }

    # Add spacing
    padding <- nchar(paste("Class:", "", "Ave"))
    cat(class, rep("", (padding + str_diff[class_pos])), paste(formatted_class_met), "\n")
    # Update variable
    class_pos <- class_pos + 1
  }
}

.split_colnames <- function(class, df) {
  class_cols <- c()
  for (colname in colnames(df)) {
    split_colname <- unlist(strsplit(colname, split = " "))
    # Remove the first and last element, corresponds to "Class:" and some metric name
    split_colname <- split_colname[-c(1, length(split_colname))]
    split_classname <- unlist(strsplit(class, split = " "))
    if (all(split_classname %in% split_colname) && length(setdiff(split_colname, split_classname)) == 0) {
      # Store colnames for the class is variable
      class_cols <- c(class_cols, colname)
    }
  }

  return(class_cols)
}

# Calculate string length of classes to create a border of dashed lines
.dashed_lines <- function(classes, return_str = FALSE) {
  str_len <- sapply(classes, nchar)
  max_str_len <- max(str_len)
  cat("\n")
  partial_output_names <- "Average Precision:  Average Recall:  Average F1:\n\n"
  cat(rep("-", nchar(paste("Class:", strrep(" ", max_str_len), partial_output_names)) %/% 1.5), "\n")
  cat("\n\n")

  if (return_str) {
    return(list("max" = max_str_len, "diff" = max_str_len - str_len))
  }
}
