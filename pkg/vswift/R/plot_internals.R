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

# Helper function for regular plotting
.visible_plots <- function(object, split, cv, model_name, model_list){
  # Check if RStudio or GUI is running for proper plotting
  new_window <- vswift:::.check_env()
  if(all(is.data.frame(object[["metrics"]][[model_name]][["split"]]), split == TRUE)){
    # Plot metrics for training and test
    new_window()
    plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n",
         main = model_list[[model_name]])
    axis(1, at = 1:2, labels = c("Training","Test"))
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      new_window()
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main = sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main = sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      
      new_window()
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("Class: %s F-Score", class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main = sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][[model_name]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][[model_name]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][[model_name]][["cv"]])[colnames(object[["metrics"]][[model_name]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][[model_name]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n", main = model_list[[model_name]])
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      } else {
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        new_window()
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = sprintf("%s - Class: %s", model_list[[model_name]], as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector, na.rm = T), col = "red", lwd = 1)
      abline(h = mean(num_vector, na.rm = T) + sd(num_vector, na.rm = T), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector, na.rm = T) - sd(num_vector, na.rm = T), col = "blue", lty = 2, lwd = 1)
    }
  }
}

.dev_off_and_new <- function(){
  # Don't display plot if save_plot is TRUE
  if(all(Sys.getenv("RStudio") == "1",rstudioapi::isAvailable())){
    graphics.off()
  } else {
    dev.off()
  }
}

.save_plots <- function(object, path, split, cv, model_name, model_list, ...){
  if(all(is.data.frame(object[["metrics"]][[model_name]][["split"]]), split == TRUE)){
    # Save metrics for training and test
    png(filename = paste0(path,sprintf("%s_train_test_classification_accuracy.png",model_list[[model_name]])), ...)
    plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n",
         main = model_list[[model_name]])
    axis(1, at = 1:2, labels = c("Training","Test"))
    vswift:::.dev_off_and_new()
    # Iterate over classes
    for(class in as.character(object[["classes"]][[1]])){
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("%s_train_test_precision_%s.png",model_list[[model_name]],class)),...)
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("%s - Class: %s Precision", model_list[[model_name]], class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
           main =  sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("%s_train_test_recall_%s.png",model_list[[model_name]],class)),...)
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("%s - Class: %s Recall", model_list[[model_name]], class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
           main =  sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
      # Save metrics for training and test
      png(filename = paste0(path,sprintf("%s_train_test_f-score_%s.png", model_list[[model_name]], class)),...)
      plot(x = 1:2, y = object[["metrics"]][[model_name]][["split"]][1:2,sprintf("%s - Class: %s F-Score", model_list[[model_name]], class)] , ylim = c(0,1), xlab = "Set", ylab = "F-Score" , xaxt = "n",
           main =  sprintf("%s - Class: %s",model_list[[model_name]],class))
      axis(1, at = 1:2, labels = c("Training","Test"))
    }
  }
  # Plot metrics for training and test
  if(all(is.data.frame(object[["metrics"]][[model_name]][["cv"]]), cv == TRUE)){
    # To get the correct class for plot title
    class_idx <- 1
    # Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
    idx <- nrow(object[["metrics"]][[model_name]][["cv"]]) - 3
    fold_n <- idx
    # Initialize new metrics
    for(colname in colnames(object[["metrics"]][[model_name]][["cv"]])[colnames(object[["metrics"]][[model_name]][["cv"]]) != "Fold"]){
      num_vector <- object[["metrics"]][[model_name]][["cv"]][1:idx, colname]
      # Split column name
      split_vector <- unlist(strsplit(colname, split = " "))
      # Depending on column name, plotting is handled slightly differently
      if("Classification" %in% split_vector){
        # Save metrics for cv
        png(filename = paste0(path,sprintf("%s_cv_classification_accuracy.png",model_list[[model_name]])),...)
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n",
             main = model_list[[model_name]])
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
      } else {
        # Get correct metric name for plot y title
        y_name <- c("Precision","Recall","F-Score")[which(c("Precision","Recall","F-Score") %in% split_vector)]
        # Save metrics for cv
        png(filename = paste0(path,sprintf("%s_cv_%s_%s.png", model_list[[model_name]],tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]])),...)
        file <- paste0(path,sprintf("%s_cv_%s_%s.png", model_list[[model_name]],tolower(y_name),as.character(object[["classes"]][[1]])[[class_idx]]))
        plot(x = 1:fold_n, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = sprintf("%s - Class: %s", model_list[[model_name]], as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
        axis(side = 1, at = as.integer(1:fold_n), labels = as.integer(1:fold_n))
        # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
        if(y_name == "F-Score"){
          class_idx <- class_idx + 1
        }
      }
      # Add mean and standard deviation to the plot
      abline(h = mean(num_vector, na.rm = T), col = "red", lwd = 1)
      abline(h = mean(num_vector, na.rm = T) + sd(num_vector, na.rm = T), col = "blue", lty = 2, lwd = 1)
      abline(h = mean(num_vector, na.rm = T) - sd(num_vector, na.rm = T), col = "blue", lty = 2, lwd = 1)
      # Don't display plot and create new plot
      vswift:::.dev_off_and_new()
    }
  }
}
