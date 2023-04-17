#Create vswift class
categorical.cv.split  <- function(data = NULL, y_col = NULL,x_col = NULL, k = NULL, split = NULL, model_type = NULL, stratified = FALSE,  random_seed = NULL,...){
  " Parameters:
      -----------
      
      `data`: A dataframe containing the response variable and predictors.
      `y_col`: The name of the response variable to be analyzed.
      `x_col`: A vector of column indices or a vector of column names indicating the predictors to be used. If `NULL`, all columns except for the response variable will be used as predictors. Default value is `NULL`.
      `k`: A non-negative integer value up to 30 for k-fold cross-validation.
      `split`: A numeric value between 0.5 and 0.9 to determine the proportion of the dataset that will be used for training.
      `model_type`: The type of algorithm to be used for data analysis. Currently supported options include logistic regression, linear discriminant analysis (LDA), quadratic discriminant analysis (QDA), support vector machine (SVM), and naive Bayes.
      `stratified`: A logical value indicating whether to use stratified sampling during data splitting. If `TRUE`, the relative proportion of categories in the response variable is maintained in the training and test datasets. Default value is `NULL`.
      `plot_metrics`: A logical value indicating whether to plot evaluation metrics - Classification Accuracy, Precision, Recall, F1. Default value is `FALSE`.
      `random_seed`: numerical value to set seed"
  #Checking if inputs are valid
  .error.handling(data = data, y_col = y_col, x_col = x_col, k = k, split = split, model_type = model_type, stratified = stratified, random_seed = random_seed, call = "categorical.cv.split")
  #Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  #Print parameter information
  cat(sprintf("Model Type: %s\n\n", model_type))
  #Creating response variable
  cat(sprintf("Response Variable: %s\n\n", response_var <- ifelse(is.character(y_col), y_col, colnames(data)[y_col])))
  #Creating feature vector
  if (is.null(x_col)){
    feature_vec <- colnames(data)[colnames(data) != response_var]
  }else {
    if(all(is.character(x_col))){
      feature_vec <- x_col
    }else{
      feature_vec <- colnames(data)[x_col]
    }
  }
  cat(sprintf("Features: %s\n\n", paste(feature_vec, collapse = ", ")))
  cat(sprintf("Classes: %s\n\n", paste(names(table(data[,response_var])), collapse = ", ")))
  cat(sprintf("K: %s\n\n", k))
  cat(sprintf("Split: %s\n\n", split))
  cat(sprintf("Stratified Sampling: %s\n\n", stratified))
  cat(sprintf("Random Seed: %s\n\n", random_seed))
  #Get response and predictors
  if(is.character(y_col)){
    y_col <- which(colnames(data) == y_col)
  }
  #Remove rows with missing data
  cleaned_data <- data[complete.cases(data),]
  #Make response a factor
  cleaned_data[,response_var] <- factor(cleaned_data[,response_var])
  #Print sample size and missing data for user transparency
  cat(sprintf("Missing Data: %s\n\n", nrow(data) - nrow(cleaned_data)))
  cat(sprintf("Sample Size: %s\n", nrow(cleaned_data)))
  #Initialize output list
  categorical.cv.split_output <- list()
  #Store classes
  categorical.cv.split_output[["classes"]][[response_var]] <- names(table(cleaned_data[,response_var]))
  #Create formula string
  categorical.cv.split_output[["formula"]] <- formula <- as.formula(paste(response_var, "~", paste(feature_vec, collapse = " + ")))
  #Get names and create a dictionary to convert to numeric if logistic model is chosen
  if(model_type == "logistic"){
    #Start at 0
    class_position <- 0
    for(class in names(table(cleaned_data[,response_var]))){
      categorical.cv.split_output[["class_dictionary"]][[as.character(class)]] <- class_position 
      class_position <- class_position  + 1
    }
  }
  if(stratified == TRUE){
    #Initialize list; initializing for ordering output purposes
    categorical.cv.split_output[["class_indices"]] <- list()
    #Get proportions
    categorical.cv.split_output[["class_proportions"]] <- table(cleaned_data[,response_var])/sum(table(cleaned_data[,response_var]))
    #Get the indices with the corresponding categories and ass to list
    for(class in names(categorical.cv.split_output[["class_proportions"]])){
      categorical.cv.split_output[["class_indices"]][[class]]  <- which(cleaned_data[,response_var] == class)
    }
  }
  #Stratified sampling
  if(!is.null(split)){
    if(stratified == TRUE){
      #Get out of .stratified.sampling
      stratified.sampling_output <- .stratified.sampling(data = cleaned_data,type = "split", split = split, output = categorical.cv.split_output, response_var = response_var, random_seed = random_seed)
      #Create training and test set
      training_data <- cleaned_data[stratified.sampling_output$output$sample_indices$training,]
      test_data <- cleaned_data[stratified.sampling_output$output$sample_indices$test,]
      #Extract updated categorical.cv.split_output output list
      categorical.cv.split_output <- stratified.sampling_output$output
    }else{
      #Create test and training set
      training_indices <- sample(1:nrow(cleaned_data),size = round(nrow(cleaned_data)*split,0),replace = F)
      training_data <- cleaned_data[training_indices,]
      test_data <- cleaned_data[-training_indices,]
      #Store indices in list
      categorical.cv.split_output[["sample_indices"]][["training"]] <- c(1:nrow(cleaned_data))[training_indices]
      categorical.cv.split_output[["sample_indices"]][["test"]] <- c(1:nrow(cleaned_data))[-training_indices]
    }
    #Create data table
    categorical.cv.split_output[["metrics"]][["split"]] <- data.frame(matrix(nrow = 2, ncol = 1))
    colnames(categorical.cv.split_output[["metrics"]][["split"]]) <- "Set"
    categorical.cv.split_output[["metrics"]][["split"]][1:2,"Set"] <- c("training","test")
  }
  #Adding information to data frame
  if(!is.null(k)){
    categorical.cv.split_output[["metrics"]][["cv"]] <- data.frame(matrix(nrow = 1,ncol = 1))
    colnames(categorical.cv.split_output[["metrics"]][["cv"]]) <- "Fold"
    #Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    #Initialize list to store fold indices; third subindex needs to be initialized
    categorical.cv.split_output[["sample_indices"]][["cv"]] <- list()
    #Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      #Initialize list to store fold proportions; third level
      categorical.cv.split_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- .stratified.sampling(data = cleaned_data, type = "k-fold", output = categorical.cv.split_output,
                                                        response_var = response_var, k = k,
                                                        random_seed = random_seed)
      #Collect output
      categorical.cv.split_output <- stratified.sampling_output$output
    }else{
      #Get floor
      fold_size_vector <- rep(floor(nrow(cleaned_data)/k),k)
      excess <- nrow(cleaned_data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:k,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      #random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:k){
        #Add name to dataframe
        categorical.cv.split_output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
        #Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        #Add indices to list
        categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
    #Reorder list
    metrics_position <- which(names(categorical.cv.split_output) == "metrics")
    categorical.cv.split_output <- c(categorical.cv.split_output[-metrics_position],categorical.cv.split_output[metrics_position])
    
  }
  #Ensure model type is lowercase
  model_type <- tolower(model_type)
  #Add it plus one to the iterator if k is not null
  iterator <- ifelse(is.null(k), 1, k + 1)
  #Initialize list to store training models
  if(!is.null(split)){
    categorical.cv.split_output[[paste0(model_type,"_models")]][["split"]] <- list()
    #Create iterator vector
    iterator_vector <- 1:iterator
  }
  if(!is.null(k)){
    categorical.cv.split_output[[paste0(model_type,"_models")]][["cv"]] <- list()
    #Create iterator vector
    if(!is.null(split)){
      #Create iterator vector
      iterator_vector <- 1:iterator
    } else{
      #Create iterator vector
      iterator_vector <- 2:iterator
    }
    
  }
  #Convert variables to characters so that models will predict the original variable
  if(model_type == "logistic"){
    cleaned_data[,response_var] <- sapply(data[,response_var], function(x) categorical.cv.split_output[["class_dictionary"]][[as.character(x)]])
    training_data[,response_var] <- sapply(training_data[,response_var], function(x) categorical.cv.split_output[["class_dictionary"]][[as.character(x)]])
    test_data[,response_var] <- sapply(test_data[,response_var], function(x) categorical.cv.split_output[["class_dictionary"]][[as.character(x)]])
  }
  #First iteration will always be the evaluation for the traditional data split method
  for(i in iterator_vector){
    if(i == 1){
      #Assigning data split matrices to new variable 
      model_data <- training_data
    }else{
      #After the first iteration the cv begins, the training set is assigned to a new variable
      model_data <- cleaned_data[-c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
      validation_data <- cleaned_data[c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
    }
    #Generate model depending on chosen model_type
    switch(model_type,
           "lda" = {model <- MASS::lda(formula, data = model_data,...)},
           "qda" = {model <- MASS::qda(formula, data = model_data,...)},
           "logistic" = {model <- glm(formula, data = model_data , family = "binomial",...)},
           "svm" = {model <- e1071::svm(formula, data = model_data,...)},
           "naivebayes" = {model <- naivebayes::naive_bayes(formula = formula, data = model_data,...)}
    )
    #Create variables used in for loops to calculate precision, recall, and f1
    switch(model_type,
           "logistic" = {
             classes <- as.numeric(unlist(categorical.cv.split_output[["class_dictionary"]]))
             class_names <- names(categorical.cv.split_output[["class_dictionary"]])
           },
           class_names <- classes <- categorical.cv.split_output[["classes"]][[response_var]])
    #Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("training","test")){
        if(j == "test"){
          #Assign test set to new variable
          model_data <- test_data
        } else{
          categorical.cv.split_output[[paste0(model_type,"_models")]][["split"]][[j]] <- model
        }
        #Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = model_data)},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = model_data, type = "response")
                 prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = model_data)},
               prediction_vector <- predict(model, newdata = model_data)$class
        )
        #Calculate classification accuracy
        categorical.cv.split_output[["metrics"]][["split"]][which(categorical.cv.split_output[["metrics"]][["split"]]$Set == j),"Classification Accuracy"] <- sum(model_data[,response_var] == prediction_vector)/length(model_data[,response_var])
        #Class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          #Sum of true positives
          true_pos <- sum(model_data[,response_var][which(model_data[,response_var] == class)] == prediction_vector[which(model_data[,response_var] == class)])
          #Sum of false negatives
          false_neg <- abs(true_pos - length(model_data[,response_var][which(model_data[,response_var]  == class)]))
          #Sum of the false positive
          false_pos <- length(which(prediction_vector[-which(model_data[,response_var] == class)] == class))
          #Add metrics to dataframe
          categorical.cv.split_output[["metrics"]][["split"]][which(categorical.cv.split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Precision", class_names[class_position])] <- true_pos/(true_pos + false_pos)
          categorical.cv.split_output[["metrics"]][["split"]][which(categorical.cv.split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Recall", class_names[class_position])] <- true_pos/(true_pos + false_neg)
          categorical.cv.split_output[["metrics"]][["split"]][which(categorical.cv.split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s F1", class_names[class_position])] <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
          class_position <- class_position + 1
        }
      }
    } else{
      if(all(!is.null(k),(i-1) <= k)){
        #Assign validation data to new variables
        model_data <- validation_data
        categorical.cv.split_output[[paste0(model_type,"_models")]][["cv"]][[sprintf("fold %s", i-1)]] <- model
        # Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = model_data)},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = model_data, type = "response")
                 prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = model_data)},
               prediction_vector <- predict(model, newdata = model_data)$class
        )
        #Calculate classification accuracy for fold
        categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- sum(model_data[,response_var] == prediction_vector)/length(model_data[,response_var])
        #Reset class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          #Sum of true positives
          true_pos <- sum(model_data[,response_var][which(model_data[,response_var] == class)] == prediction_vector[which(model_data[,response_var] == class)])
          #Sum of false negatives
          false_neg <- abs(true_pos  - length(model_data[,response_var][which(model_data[,response_var]  == class)]))
          #Sum of the false positive
          false_pos <- length(which(prediction_vector[-which(model_data[,response_var] == class)] == class))
          #Add metrics to dataframe
          categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Precision", class_names[class_position])] <- true_pos/(true_pos + false_pos)
          categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Recall", class_names[class_position])] <- true_pos/(true_pos + false_neg)
          categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s F1", class_names[class_position])] <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
          class_position <- class_position + 1
        }
      }
    }
    #Calculate mean, standard deviation, and standard error for cross validation
    if(all(!is.null(k),(i-1) == k)){
      idx <- nrow(categorical.cv.split_output[["metrics"]][["cv"]] )
      categorical.cv.split_output[["metrics"]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
      #Calculate mean, standard deviation, and sd for each column except for fold
      for(colname in colnames(categorical.cv.split_output[["metrics"]][["cv"]] )[colnames(categorical.cv.split_output[["metrics"]][["cv"]] ) != "Fold"]){
        #Create vector containing corresponding column name values for each fold
        num_vector <- categorical.cv.split_output[["metrics"]][["cv"]] [1:idx, colname]
        categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
        categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
        categorical.cv.split_output[["metrics"]][["cv"]][which(categorical.cv.split_output[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(k)
      }
    }
    }
  #Make list a vswift class
  class(categorical.cv.split_output) <- "vswift"
  return(categorical.cv.split_output)
}
