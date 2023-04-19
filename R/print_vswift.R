"print.vswift"<- function(object, parameters = TRUE, performance = TRUE){
  if(class(object) == "vswift"){
    if(parameters == TRUE){
      #Print parameter information
      cat(sprintf("Model Type: %s\n\n", object[["information"]][["parameters"]][["model_type"]]))
      #Creating response variable
      cat(sprintf("Features: %s\n\n", paste(object[["information"]][["parameters"]][["features"]], collapse = ",")))
      cat(sprintf("Response variable: %s\n\n", object[["information"]][["parameters"]][["response_variable"]]))
      cat(sprintf("Classes: %s\n\n", paste(unlist(object[["classes"]]), collapse = ", ")))
      cat(sprintf("K: %s\n\n", object[["information"]][["parameters"]][["k"]]))
      cat(sprintf("Split: %s\n\n", object[["information"]][["parameters"]][["split"]]))
      cat(sprintf("Stratified Sampling: %s\n\n", object[["information"]][["parameters"]][["stratified"]]))
      cat(sprintf("Random Seed: %s\n\n", object[["information"]][["parameters"]][["random_seed"]]))
      #Print sample size and missing data for user transparency
      cat(sprintf("Missing Data: %s\n\n", object[["information"]][["parameters"]][["missing_data"]]))
      cat(sprintf("Sample Size: %s\n\n", object[["information"]][["parameters"]][["sample_size"]]))
    }
  }
  #Print performance
  if(performance == TRUE){
    #Calculate string length of classes
    string_length <- sapply(unlist(object["classes"]), function(x) nchar(x))
    max_string_length <- max(string_length)
    string_diff <- max_string_length - string_length 
    #Print performance metrics for train test set if the dataframe exists
    if(is.data.frame(object[["metrics"]][["split"]])){
      for(set in c("Training","Test")){
        #Variable for which class string length to print to ensure all values have equal spacing
        class_position <- 1 
        #Print name of the set metrics to be printed and add undercores
        cat("\n\n",set,"\n")
        cat(rep("_",nchar(set)),"\n\n")
        #Print classification accuracy
        cat("Classication Accuracy: ", format(round(object[["metrics"]][["split"]][which(object[["metrics"]][["split"]]$Set == set),"Classification Accuracy"],2), nsmall = 2),"\n\n")
        #Print name of metrics
        diff <- abs(nchar("Class:") - max_string_length)
        cat("Class:",rep("", max_string_length),"Precision:  Recall:  F-Score:\n\n")
        #For loop to obtain vector of values for each class
        for(class in unlist(object["classes"])){
          #Create empty variable
          class_col<- c()
          #Go through column names, split the colnames and class name to see if the column name is the metric for that class
          for(colname in colnames(object[["metrics"]][["split"]])){
            split_colname <- unlist(strsplit(colname,split = " "))
            split_classname <- unlist(strsplit(class,split = " ")) 
            if(all(split_classname %in% split_colname)){
              #Store colnames for the class is variable
              class_col <- c(class_col, colname)
            }
          }
          #Print metric corresponding to class
          class_metrics <- sapply(object[["metrics"]][["split"]][which(object[["metrics"]][["split"]]$Set == "Training"),class_col], function(x) format(round(x,2), nsmall = 2))  
          #Add spacing
          padding <- nchar(paste("Class:",rep("", max_string_length),"Pre"))[1]
          cat(class,rep("",(padding + string_diff[class_position])),paste(class_metrics, collapse = " "),"\n")
          class_position <- class_position + 1
        }
        
      }
      
    }
    if(all(is.data.frame(object[["metrics"]][["cv"]]))){
      class_position <- 1 
      k <- object[["information"]][["parameters"]][["k"]]
      cat("\n\n","K-fold CV","\n")
      cat(rep("_",nchar("K-fold CV")),"\n\n")
      cat("Average Classication Accuracy: ", format(round(object[["metrics"]][["cv"]][which(object[["metrics"]][["cv"]]$Fold == "Mean CV:"),"Classification Accuracy"],2), nsmall = 2),"\n\n")
      cat("Class:",rep("", 12),"Average Precision:  StDev Precision:  Average Recall:  StDev Recall:  Average F-score:  StDev F-score:\n\n")
      for(class in unlist(object["classes"])){
        class_col <- c()
        for(colname in colnames(object[["metrics"]][["split"]])){
          split_colname <- unlist(strsplit(colname,split = " "))
          split_classname <- unlist(strsplit(class,split = " ")) 
          if(all(split_classname %in% split_colname)){
            class_col <- c(class_col, colname)
          }
        }
        mean_class_metrics <- sapply(object[["metrics"]][["cv"]][((k+1)),class_col], function(x) format(round(x,2), nsmall = 2))  
        sd_class_metrics <- sapply(object[["metrics"]][["cv"]][((k+2)),class_col], function(x) format(round(x,2), nsmall = 2))  
        class_metrics <- c(mean_class_metrics[1],sd_class_metrics[1],mean_class_metrics[2],sd_class_metrics[2],
                           mean_class_metrics[3],sd_class_metrics[3])
        #cat(class,rep("",(15 + string_diff[class_position])),paste(class_metrics, collapse = "   "),"\n")
        cat(class,rep("",(15 + string_diff[class_position])),paste(class_metrics),"\n")
        class_position <- class_position + 1
      }
      
    }
  }
}