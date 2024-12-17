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
  info <- x$configs$model_params

  if (is.null(info$logistic_threshold)) info <- info[!names(info) == "logistic_threshold"]

  info$map_args <- info$map_args[!names(info$map_args) != model]

  if (length(info$map_args) == 0) {
    info$map_args <- NULL
    info <- c(list(map_args = NULL), info)
  }

  str <- capture.output(dput(info))
  str <- gsub("\\s+", " ", paste(str, collapse = ""))
  cat(sprintf("Model Parameters: %s\n\n", str))

  # Print sample size and missing data for user transparency
  cat(sprintf("Unlabeled Data: %s\n\n", x$missing_data_summary$unlabeled_data))
  cat(sprintf("Incomplete Labeled Data: %s\n\n", x$missing_data_summary$incomplete_labeled_data))
  if (!is.null(x$configs$impute_params$method)) {
    total <- x$missing_data_summary$complete_data + x$missing_data_summary$incomplete_labeled_data
    cat(sprintf("Sample Size (Complete + Imputed Incomplete Labeled Data): %s\n\n", total))
  } else {
    cat(sprintf("Sample Size (Complete Data): %s\n\n", x$missing_data_summary$complete_data))
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
      # Empty class_col or initialize variable
      class_col <- c()

      # Go through column names, split the colnames and class name to see if the column name is the metric for that class
      for (colname in colnames(df)) {
        split_colname <- unlist(strsplit(colname, split = " "))
        split_classname <- unlist(strsplit(class, split = " "))
        if (all(split_classname %in% split_colname)) {
          # Store colnames for the class is variable
          class_col <- c(class_col, colname)
        }
      }

      # Print metric corresponding to class
      class_met <- sapply(df[df$Set == set, class_col], function(x) format(round(x, 2), nsmall = 2))
      # Add spacing
      padding <- nchar(paste("Class:", "", "Pre"))

      if (class_met[1] == "NaN") {
        class_met <- c(class_met[1], rep("", 5), class_met[2], rep("", 5), class_met[3])
      } else {
        class_met <- c(class_met[1], rep("", 4), class_met[2], rep("", 5), class_met[3])
      }

      cat(class, rep("", (padding + str_diff[class_pos])), paste(class_met, collapse = " "), "\n")
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
    # Empty class_col or initialize variable
    class_col <- c()

    for (colname in colnames(df)) {
      split_colname <- unlist(strsplit(colname, split = " "))
      split_classname <- unlist(strsplit(class, split = " "))
      if (all(split_classname %in% split_colname)) {
        # Store colnames for the class is variable
        class_col <- c(class_col, colname)
      }
    }

    # Print metric corresponding to class
    mean_met <- sapply(df[((n_folds + 1)), class_col], function(x) format(round(x, 2), nsmall = 2))
    sd_met <- sapply(df[((n_folds + 2)), class_col], function(x) format(round(x, 2), nsmall = 2))
    sd_met_pos <- 1
    class_met <- c()

    for (metric in mean_met) {
      class_met <- c(class_met, sprintf("%s \U00B1 %s (SD)", metric, sd_met[sd_met_pos]))
      sd_met_pos <- sd_met_pos + 1
    }

    if (class_met[1] == "NaN (NA)") {
      class_met <- c(rep("", 3), class_met[1], rep("", 6), class_met[2], rep("", 6), class_met[3])
    } else {
      class_met <- c(class_met[1], rep("", 6), class_met[2], rep("", 6), class_met[3])
    }

    # Add spacing
    padding <- nchar(paste("Class:", "", "Ave"))
    cat(class, rep("", (padding + str_diff[class_pos])), paste(class_met), "\n")
    # Update variable
    class_pos <- class_pos + 1
  }
}

# Calculate string length of classes to create a border of dashed lines
.dashed_lines <- function(classes, return_str = FALSE) {
  str_len <- sapply(classes, function(x) nchar(x))
  max_str_len <- max(str_len)
  cat("\n")
  partial_output_names <- "Average Precision:  Average Recall:  Average F1:\n\n"
  cat(rep("-", nchar(paste("Class:", strrep(" ", max_str_len), partial_output_names)) %/% 1.5), "\n")
  cat("\n\n")

  if (return_str) {
    return(list("max" = max_str_len, "diff" = max_str_len - str_len))
  }
}
