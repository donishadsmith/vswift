# Detect current environment for proper plotting
.display <- function(path) {
  if (is.null(path)) {
    if (Sys.getenv("RStudio") == "0") dev.new()
  } else {
    if (Sys.getenv("RStudio") == "1") {
      graphics.off()
    } else {
      dev.off()
    }
  }
}

# Entry point for plotting train-test split and cross-validation evaluation metrics
.plot <- function(x, metrics, model, plot_title, split, cv, class_names, path, ...) {
  # Get dataframe
  df <- x$metrics[[model]]

  if (!is.null(path)) {
    # Get OS separator
    os.sep <- ifelse(as.vector(Sys.info()["sysname"] == "Windows"), "\\", "/")
    # Get name for png by replacing whitespace with underscores
    png_name <- strsplit(plot_title, split = " ") |>
      unlist() |>
      paste(collapse = "_") |>
      tolower()
  }

  # Create Metrics List
  metrics_list <- list("precision" = "Precision", "recall" = "Recall", "f1" = "F1")

  # Get classes
  if (is.null(class_names)) {
    classes <- x$class_summary$classes
  } else {
    classes <- class_names
  }

  # Plot train-test split evaluation metrics
  if (all(is.data.frame(df$split), isTRUE(split))) {
    .plot_split(df, classes, metrics, metrics_list, plot_title, path,
      os.sep = if (exists("os.sep")) os.sep else NULL,
      png_name = if (exists("png_name")) png_name else NULL,
      ...
    )
  }

  # Plot cross-validation evaluation metrics
  if (all(is.data.frame(df$cv), isTRUE(cv))) {
    .plot_cv(df, classes, metrics, metrics_list, plot_title, path,
      os.sep = if (exists("os.sep")) os.sep else NULL,
      png_name = if (exists("png_name")) png_name else NULL, ...
    )
  }
}

# Function to plot train-test split evaluation metrics
.plot_split <- function(df, classes, metrics, metrics_list, plot_title, path, os.sep, png_name, ...) {
  # Base plot kwargs
  plot_kwargs <- list(x = 1:2, ylim = 0:1, xlab = "Set", xaxt = "n")
  axis_kwargs <- list(side = 1, at = 1:2, labels = c("Training", "Test"))

  # Plot metrics for training and test
  if ("accuracy" %in% metrics) {
    # Add additional keys
    plot_kwargs$y <- df$split[1:2, "Classification Accuracy"]
    plot_kwargs$ylab <- "Classification Accuracy"
    plot_kwargs$main <- plot_title

    # Create png
    if (!is.null(path)) {
      png(
        filename = paste0(path, os.sep, sprintf("%s_train_test_classification_accuracy.png", png_name)),
        ...
      )
    }

    # Plot data
    do.call(plot, plot_kwargs)
    # Add axis info
    do.call(axis, axis_kwargs)
    # Remove accuracy
    metrics <- metrics[metrics != "accuracy"]
    # Use dev.new for certain R environments or dev.off if png is used
    .display(path)
  }

  # Reduce nesting by creating empty vector
  metrics <- if (length(metrics) == 0) c() else metrics

  # Plot metrics for training and test; Iterate over classes
  for (class in classes) {
    for (metric in metrics) {
      # Add additional keys
      plot_kwargs$y <- df$split[1:2, sprintf("Class: %s %s", class, metrics_list[[metric]])]
      plot_kwargs$ylab <- metrics_list[[metric]]
      plot_kwargs$main <- sprintf("%s - Class: %s", plot_title, class)

      # Create png
      if (!is.null(path)) {
        png(filename = paste0(
          path, os.sep,
          sprintf("%s_train_test_%s_%s.png", png_name, metric, paste(unlist(strsplit(class, split = " ")), collapse = "_"))
        ), ...)
      }

      # Plot data
      do.call(plot, plot_kwargs)
      # Add axis info
      do.call(axis, axis_kwargs)
      # Use dev.new for certain R environments or dev.off if png is used
      .display(path)
    }
  }
}

# Function to plot cross-validation evaluation metrics
.plot_cv <- function(df, classes, metrics, metrics_list, plot_title, path, os.sep, png_name, ...) {
  # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
  idx <- nrow(df$cv) - 3
  # Create vector of metrics to obtain
  col_names <- c()

  # Populate col_names
  if ("accuracy" %in% metrics) col_names <- c("Classification Accuracy")

  if (any(names(metrics_list) %in% metrics)) {
    # Intersect metrics and convert names
    intersected_metrics <- intersect(metrics, names(metrics_list))
    converted_metrics <- lapply(intersected_metrics, function(x) metrics_list[[x]])
    # Get column names from dataframe
    col_names <- c(col_names, as.vector(sapply(classes, function(x) paste("Class:", x, converted_metrics))))
  }

  # Base plot kwargs
  plot_kwargs <- list(x = 1:idx, ylim = c(0, 1), xlab = "Folds", xaxt = "n")

  for (col_name in col_names) {
    # Get values
    plot_kwargs$y <- df$cv[1:idx, col_name]

    # Get Title and
    if (col_name == "Classification Accuracy") {
      # Get ylab and main
      plot_kwargs$ylab <- "Classification Accuracy"
      plot_kwargs$main <- plot_title

      if (!is.null(path)) {
        full_png_name <- sprintf("%s_cv_classification_accuracy.png", png_name)
        filename <- paste0(path, os.sep, full_png_name)
      }
    } else {
      # Get name of metric - "Precision", "Recall", "F1"
      split_vec <- unlist(strsplit(col_name, split = " "))
      # Final character in vector is the metric name
      metric_name <- split_vec[length(split_vec)]
      # Get class name; Exclude "Class" and the metric name to get class name
      class_name <- paste(split_vec[-which(split_vec %in% c("Class:", metric_name))],
        collapse = " "
      )

      # Get ylab and main
      plot_kwargs$ylab <- metric_name
      plot_kwargs$main <- sprintf("%s - Class: %s", plot_title, class_name)

      if (!is.null(path)) {
        full_png_name <- sprintf("%s_cv_%s_%s.png", png_name, metric_name, paste(class_name, collapse = "_"))
        filename <- paste0(path, os.sep, full_png_name)
      }
    }

    # Create png
    if (!is.null(path)) png(filename = filename, ...)

    # Generate plot
    do.call(plot, plot_kwargs)
    # Add axis info
    axis(side = 1, at = as.integer(1:idx), labels = as.integer(1:idx))
    # Add mean and standard deviation to the plot
    abline(h = mean(plot_kwargs$y, na.rm = TRUE), col = "red", lwd = 1)
    abline(h = mean(plot_kwargs$y, na.rm = TRUE) + sd(plot_kwargs$y, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)
    abline(h = mean(plot_kwargs$y, na.rm = TRUE) - sd(plot_kwargs$y, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)

    # Add legend
    legend("bottomright", legend = c("Mean", "Mean \U00B1 SD"), col = c("red", "blue"), lty = c(1, 2), lwd = 1)

    # Use dev.new for certain R environments or dev.off if png is used
    .display(path)
  }
}
