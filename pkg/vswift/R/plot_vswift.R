#' Plot model evaluation metrics
#' 
#' `plot.vswift` plots model evaluation metrics (classification accuracy and precision, recall, and f-score for each class) from a vswift object. 
#'
#'
#' @param object An object of class vswift.
#' @param split A logical value indicating whether to plot metrics for train-test splitting results. Default = TRUE.
#' @param cv A logical value indicating whether to plot metrics for k-fold cross-validation results. Note: Solid blueline represents the mean
#' and dashed red line represents the standard deviation. Default = TRUE.
#' 
#' @return Plots representing evaluation metrics.
#' @examples
#' # Load an example dataset
#' 
#' data(iris)
#' 
#' # Perform a train-test split with an 80% training set and stratified_sampling using QDA
#' 
#' result <- categorical_cv_split(data = iris, target = "Species", split = 0.8,
#' model_type = "qda", stratified = TRUE)
#' 
#' # Plot performance metrics for train-test split
#' 
#' plot(result)
#' 
#' @export
"plot.vswift" <- function(object, split = TRUE, cv = TRUE){
  if(class(object) == "vswift"){
    # Check if RStudio or GUI is running for proper plotting
    if(Sys.getenv("RStudio") == "1"){
      new_window  <- rstudioapi::viewer()
    }else{
      system = as.character(Sys.info()["sysname"])
      new_window <- ifelse(system == "Windows", windows, x11)
    }
    if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
      # Plot metrics for training and test
      as.character(Sys.info()["sysname"])
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
        }else{
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
        abline(h = mean(num_vector) + sd(num_vector)/sqrt(fold_n), col = "blue", lty = 2, lwd = 1)
        abline(h = mean(num_vector) - sd(num_vector)/sqrt(fold_n), col = "blue", lty = 2, lwd = 1)
      }
    }
  } else{
    stop("object must be of class 'vswift'")
  }
}
