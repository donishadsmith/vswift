# Helper function to perform proper plotting depending on where command is ran
.check_env <- function(){
  system = as.character(Sys.info()["sysname"])
  if(Sys.getenv("RStudio") == "1"){
    new_window  <- ifelse(rstudioapi::isAvailable(), function(){placeholder = "placeholder"},ifelse(system == "Windows", windows, x11))
  } else {
    new_window <- ifelse(system == "Windows", windows, x11)
  }
  return(new_window)
}

.dev_off_and_new <- function(){
  # Don't display plot if save_plot is TRUE
  if(all(Sys.getenv("RStudio") == "1",rstudioapi::isAvailable())){
    graphics.off()
  } else {
    dev.off()
  }
}

# Helper function for regular plotting
.visible_plots <- function(object, split, cv){
  # Check if RStudio or GUI is running for proper plotting
  new_window <- vswift:::.check_env()
  if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
    # Plot metrics for training and test
    new_window()
    plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
    axis(1, at = 1:2, labels = c("Training","Test"))
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s F-Score", class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][["cv"]])[colnames(object[["metrics"]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      } else {
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Class: ",as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector), col = "red", lwd = 1)
      abline(h = mean(num_vector) + sd(num_vector), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector) - sd(num_vector), col = "blue", lty = 2, lwd = 1)
    }
  }
}


.save_plots <- function(object, path, split, cv, ...){
  if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
    # Save metrics for training and test
    png(filename = paste0(path,"train_test_classification_accuracy.png"),...)
    plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
    axis(1, at = 1:2, labels = c("Training","Test"))
    vswift:::.dev_off_and_new()
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_precision_%s.png",class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_recall_%s.png",class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("train_test_f-score_%s.png",  class)),...)
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s F-Score", class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main = paste("Class:",class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][["cv"]])[colnames(object[["metrics"]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        # Save metrics for cv
        png(filename = paste0(path,"cv_classification_accuracy.png"),...)
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      } else {
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        # Save metrics for cv
        png(filename = paste0(path,sprintf("cv_%s_%s.png",tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]])),...)
        file <- paste0(path,sprintf("cv_%s_%s.png",tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]]))
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Class: ",as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector), col = "red", lwd = 1)
      abline(h = mean(num_vector) + sd(num_vector), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector) - sd(num_vector), col = "blue", lty = 2, lwd = 1)
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
    }
  }
}

