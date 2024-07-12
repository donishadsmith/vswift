# Helper function to perform proper plotting depending on where command is ran
#' @noRd
#' @export
.check_env <- function(){
  system = as.character(Sys.info()["sysname"])
  if(Sys.getenv("RStudio") == "1"){
    new_window  <- ifelse(Sys.getenv("RStudio") == "1", function(){placeholder = "placeholder"}, dev.new)
  } else {
    new_window <- dev.new
  }
  return(new_window)
}

# Helper function for regular plotting
#' @noRd
#' @export
.visible_plots <- function(x, split, cv, metrics, class_names, model_name, model_list){
  # Check if RStudio or GUI is running for proper plotting
  new_window <- .check_env()
  # Simplify parameters
  df <- x[["metrics"]][[model_name]]
  # Model name
  converted_model_name_plot <- model_list[[model_name]]
  # Metrics list
  metrics_list <- list("precision" = "Precision", "recall" = "Recall", "f1" = "F-Score")
  specified_metrics <- lapply(metrics[metrics != "accuracy"], function(x) metrics_list[[x]])
  # Get classes
  if(is.null(class_names)){
    classes <- as.character(x[["classes"]][[1]])
  } else{
    classes <- class_names
  }

  if(all(is.data.frame(df[["split"]]), split == TRUE)){
    if("accuracy" %in% metrics){
      # Plot metrics for training and test
      new_window()
      # Plot data
      plot(x = 1:2, y = df[["split"]][1:2,"Classification Accuracy"],
           ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy",
           xaxt = "n", main = converted_model_name_plot)
      # Add axis info
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
    # Iterate over classes
    if(any(names(metrics_list) %in% metrics)){
      for(class in classes){
        for(metric in specified_metrics){
          # Plot metrics for training and test
          new_window()
          # Plot data
          plot(x = 1:2, y = df[["split"]][1:2,sprintf("Class: %s %s", class, metric)],
               ylim = c(0,1), xlab = "Set",ylab = metric, xaxt = "n",
               main = sprintf("%s - Class: %s", converted_model_name_plot, class))
          # Add axis info
          axis(1, at = 1:2, labels = c("Training","Test"))
        }
      }
    }
  }
  if(all(is.data.frame(df[["cv"]]), cv == TRUE)){
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(x[["metrics"]][[model_name]][["cv"]]) - 3
    # Create vector of metrics to obtain
    col_names <- c()
    if("accuracy" %in% metrics) col_names <- c("Classification Accuracy")
    if(any(names(metrics_list) %in% metrics)) col_names <- c(col_names,paste("Class:", classes, specified_metrics))

    for(col_name in col_names){
      # Get values
      num_vector <- df[["cv"]][1:idx, col_name]
      # Create png
      if(col_name == "Classification Accuracy"){
        # Get ylab and main
        ylab <- "Classification Accuracy"
        main <- converted_model_name_plot
      } else if(any(names(metrics_list) %in% metrics)){
          # Get name of metric - "Precision", "Recall", "F-Score
          split_name <- unlist(strsplit(col_name, split = " "))
          split_metric_name <- split_name[length(split_name)]
          # Get class name
          split_class_name_plot <- paste(split_name[-which(split_name %in% c("Class:", split_metric_name))], collapse = " ")
          # Get ylab and main
          ylab <- split_metric_name
          main <- sprintf("%s - Class: %s", converted_model_name_plot, split_class_name_plot)
      }
      # Plot metrics for training and test
      new_window()
      # Generate plot
      plot(x = 1:idx, y = num_vector, ylim = c(0,1), xlab = "K-folds",
           ylab = ylab, xaxt = "n", main = main)
      # Add axis info
      axis(side = 1, at = as.integer(1:idx), labels = as.integer(1:idx))
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector, na.rm = TRUE), col = "red", lwd = 1)
      abline(h = mean(num_vector, na.rm = TRUE) + sd(num_vector, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector, na.rm = TRUE) - sd(num_vector, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)
    }
  }
}

# Detect current environment for proper plotting
#' @noRd
#' @export
.dev_off_and_new <- function(){
  # Don't display plot if save_plot is TRUE
  if(Sys.getenv("RStudio") == "1"){
    graphics.off()
  } else {
    dev.off()
  }
}

# Function for plot saving
#' @noRd
#' @export
.save_plots <- function(x, path, split, cv, metrics, class_names, model_name, model_list, ...){
  # Simplify parameters
  df <- x[["metrics"]][[model_name]]
  # Model name
  converted_model_name_plot <- model_list[[model_name]]
  converted_model_name_png <- paste(unlist(strsplit(model_list[[model_name]], split = " ")), collapse = "_")
  # Metrics list
  metrics_list <- list("precision" = "Precision", "recall" = "Recall", "f1" = "F-Score")
  specified_metrics <- lapply(metrics[metrics != "accuracy"], function(x) metrics_list[[x]])
  # Get classes
  if(is.null(class_names)){
    classes <- as.character(x[["classes"]][[1]])
  } else{
    classes <- class_names
  }

  if(all(is.data.frame(df[["split"]]), split == TRUE)){
    if("accuracy" %in% metrics){
      # Create png
      png(filename = paste0(path, sprintf("%s_train_test_classification_accuracy.png",
                                         tolower(converted_model_name_png))), ...)
      # Plot data
      plot(x = 1:2, y = df[["split"]][1:2,"Classification Accuracy"],
           ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy",
           xaxt = "n", main = converted_model_name_plot)
      # Add axis info
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      .dev_off_and_new()
    }
    # Iterate over classes
    if(any(names(metrics_list) %in% metrics)){
      for(class in classes){
        for(metric in specified_metrics){
          # Create png
          png(filename = paste0(path, sprintf("%s_train_test_%s_%s.png",
                                              tolower(converted_model_name_png),
                                              tolower(metric), paste(unlist(strsplit(class, split = " ")), collapse = "_"))), ...)
          # Plot data
          plot(x = 1:2, y = df[["split"]][1:2,sprintf("Class: %s %s", class, metric)],
               ylim = c(0,1), xlab = "Set",ylab = metric, xaxt = "n",
               main = sprintf("%s - Class: %s", converted_model_name_plot, class))
          # Add axis info
          axis(1, at = 1:2, labels = c("Training","Test"))
          # Don't display plot and create new plot
          .dev_off_and_new()
        }
      }
    }
  }
  if(all(is.data.frame(df[["cv"]]), cv == TRUE)){
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(x[["metrics"]][[model_name]][["cv"]]) - 3
    # Create vector of metrics to obtain
    col_names <- c()
    if("accuracy" %in% metrics) col_names <- c("Classification Accuracy")
    if(any(names(metrics_list) %in% metrics)) col_names <- c(col_names,paste("Class:", classes, specified_metrics))

    for(col_name in col_names){
      # Get values
      num_vector <- df[["cv"]][1:idx, col_name]
      # Create png
      if(col_name == "Classification Accuracy"){
        png(filename = paste0(path, sprintf("%s_cv_classification_accuracy.png", tolower(converted_model_name_png))),...)
        # Get ylab and main
        ylab <- "Classification Accuracy"
        main <- converted_model_name_plot
      } else if(any(names(metrics_list) %in% metrics)){
          # Get name of metric - "Precision", "Recall", "F-Score
          split_name <- unlist(strsplit(col_name, split = " "))
          split_metric_name <- split_name[length(split_name)]
          # Get class name
          split_class_name_plot <- paste(split_name[-which(split_name %in% c("Class:", split_metric_name))], collapse = " ")
          split_class_name_png <- paste(split_name[-which(split_name %in% c("Class:", split_metric_name))], collapse = "_")
          # Create png
          png(filename = paste0(path, sprintf("%s_cv_%s_%s.png", tolower(converted_model_name_png),
                                             tolower(split_metric_name),split_class_name_png)),...)
          # Get ylab and main
          ylab <- split_metric_name
          main <- sprintf("%s - Class: %s", converted_model_name_plot, split_class_name_plot)
      }
      # Generate plot
      plot(x = 1:idx, y = num_vector, ylim = c(0,1), xlab = "K-folds",
           ylab = ylab, xaxt = "n", main = main)
      # Add axis info
      axis(side = 1, at = as.integer(1:idx), labels = as.integer(1:idx))
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector, na.rm = TRUE), col = "red", lwd = 1)
      abline(h = mean(num_vector, na.rm = TRUE) + sd(num_vector, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector, na.rm = TRUE) - sd(num_vector, na.rm = TRUE), col = "blue", lty = 2, lwd = 1)
      # Don't display plot and create new plot
      .dev_off_and_new()
    }
  }
}
