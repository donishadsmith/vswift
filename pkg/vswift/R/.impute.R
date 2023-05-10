# Helper function for imputation
.impute <- function(data = NULL, missing_columns = NULL, impute_method = NULL, impute_args = NULL){
  # Create empty list to store information
  missing_information <- list()
  # Get missing column information
  for(col in missing_columns){
    missing_information[[names(data)[col]]][["missing"]] <- length(which(is.na(data[,col])))
  }
  # switch statement
  switch(impute_method,
         "simple" = {
           for(col in missing_columns){
             if(is.character(data[,col]) || is.factor(data[,col])){
               frequent_class <- names(which.max(table(data[,col])))
               missing_information[[names(data)[col]]][["mode"]] <- frequent_class
               data[which(is.na(data[,col])),col]  <- frequent_class
             }
             else{
               # Check distribution
               missing_information[[names(data)[col]]][["shapiro_p.value"]] <- shapiro_p.value <- shapiro.test(data[,col])[["p.value"]]
               # If less than 0.5, distribution is not normal median will be used
               if(shapiro_p.value < 0.05){
                 missing_information[[names(data)[col]]][["median"]] <- data[which(is.na(data[,col])),col] <- median(data[,col], na.rm = TRUE)
               } else {
                 missing_information[[names(data)[col]]][["mean"]] <- data[which(is.na(data[,col])),col] <- mean(data[,col], na.rm = TRUE)
               }
             }
           }},
         "missforest" = {
           if(!is.null(impute_args)){
             impute_args[["xmis"]] <- data
             missforest_output <- do.call(missForest::missForest, impute_args)
           } else {
             missforest_output <- missForest::missForest(data)
           }
           missing_information[["missForest"]] <- missforest_output
           data <- missforest_output[["ximp"]]
         })
  
  impute_output <- list("preprocessed_data" = data, "impute_info" = missing_information)
  return(impute_output)
}