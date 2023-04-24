"plot.vswift" <- function(object, split = TRUE, cv = TRUE){
  if(class(object) == "vswift"){
    if(all(is.data.frame(object[["metrics"]][["split"]]), split == TRUE)){
      #Plot metrics for training and test
      plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
      axis(1, at = 1:2, labels = c("training","test"))
      #Iterate over classes
      for(class in as.character(object[["classes"]][[1]])){
        plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Precision", class)] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
             main = paste("Class:",class))
        axis(1, at = 1:2, labels = c("Training","Test"))
        
        plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s Recall", class)] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
             main = paste("Class:",class))
        axis(1, at = 1:2, labels = c("Training","Test"))
        
        plot(x = 1:2, y = object[["metrics"]][["split"]][1:2,sprintf("Class: %s F1", class)] , ylim = c(0,1), xlab = "Set", ylab = "F1" , xaxt = "n",
             main = paste("Class:",class))
        axis(1, at = 1:2, labels = c("Training","Test"))
    }
    }
    # Plot metrics for training and test
    if(all(is.data.frame(object[["metrics"]][["cv"]]), cv == TRUE)){
      #To get the correct class for plot title
      class_idx <- 1
      #Get the last row index subtracted by three to avoid getting mean, standard dev, and standard error
      idx <- nrow(object[["metrics"]][["cv"]]) - 3
      k <- idx
      #Initialize new metrics
      for(colname in colnames(object[["metrics"]][["cv"]])[colnames(object[["metrics"]][["cv"]]) != "Fold"]){
        num_vector <- object[["metrics"]][["cv"]][1:idx, colname]
        #Split column name
        split_vector <- unlist(strsplit(colname, split = " "))
        # depending on column name, plotting is handled slightly differently
        if("Classification" %in% split_vector){
          plot(x = 1:k, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
          axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
        }else{
          # Get correct metric name for plot y title
          y_name <- c("Precision","Recall","F1")[which(c("Precision","Recall","F1") %in% split_vector)]
          plot(x = 1:k, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Class: ",as.character(object[["classes"]][[1]])[[class_idx]]), xaxt = "n") 
          axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
          # Add 1 to `class_idx` when `y_name == "Recall"` to get correct class plot title
          if(y_name == "F1"){
            class_idx <- class_idx + 1
          }
        }
        #Add mean and standard deviation to the plot
        abline(h = mean(num_vector), col = "red", lwd = 1)
        abline(h = mean(num_vector) + sd(num_vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
        abline(h = mean(num_vector) - sd(num_vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
      }
    }
  } else{
    stop("object must be of class 'vswift'")
  }
}
