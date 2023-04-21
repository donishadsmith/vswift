categorical_cv_split  <- function(data = NULL, y_col = NULL, x_col = NULL, k = NULL, split = NULL, model_type = NULL, stratified = FALSE,  random_seed = NULL, remove_untrained_observation = FALSE, save_models = FALSE, save_data = FALSE,...){
  #Checking if inputs are valid
  .error_handling(data = data, y_col = y_col, x_col = x_col, k = k, split = split, model_type = model_type, stratified = stratified, random_seed = random_seed, call = "categorical.cv.split")
  #Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  #Creating response variable
  response_var <- ifelse(is.character(y_col), y_col, colnames(data)[y_col])
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
  #Get response and predictors
  if(is.character(y_col)){
    y_col <- which(colnames(data) == y_col)
  }
  #Remove rows with missing data
  cleaned_data <- data[complete.cases(data),]
  if(model_type == "svm"){
    #Turn to factor and get levels
    data_levels <- list()
    if(class(cleaned_data[,response_var]) == "factor"){
      data_levels[[response_var]] <- levels(cleaned_data[,response_var])
    }else{
      #Turn response column to character
      cleaned_data[,response_var] <- as.character(cleaned_data[,response_var])
    }
    #Get character columns
    character_columns <- sapply(cleaned_data,function(x) is.character(x))
    factor_columns <- sapply(cleaned_data,function(x) is.factor(x))
    columns <- c(character_columns,factor_columns) 
    #Get character column names
    columns <- colnames(cleaned_data)[columns]
    for(col in columns){
      if(is.character(cleaned_data[,col])){
        cleaned_data[,col] <- factor(cleaned_data[,col])
      }
      data_levels[[col]] <- levels(cleaned_data[,col])
    }
  }
  if(!model_type %in% c("svm","logistic")){
    cleaned_data[,response_var] <- factor(cleaned_data[,response_var])
  }
  #Initialize output list
  categorical_cv_split_output <- list()
  categorical_cv_split_output[["information"]][["analysis_type"]] <- "classification"
  categorical_cv_split_output[["information"]][["parameters"]] <- list()
  categorical_cv_split_output[["information"]][["parameters"]][["features"]] <- feature_vec
  categorical_cv_split_output[["information"]][["parameters"]][["response_variable"]]  <- response_var
  categorical_cv_split_output[["information"]][["parameters"]][["model_type"]] <- model_type
  categorical_cv_split_output[["information"]][["parameters"]][["k"]]  <- k
  categorical_cv_split_output[["information"]][["parameters"]][["stratified"]]  <- stratified
  categorical_cv_split_output[["information"]][["parameters"]][["split"]]  <- split
  categorical_cv_split_output[["information"]][["parameters"]][["random_seed"]]  <- random_seed
  categorical_cv_split_output[["information"]][["parameters"]][["missing_data"]]  <- nrow(data) - nrow(cleaned_data)
  categorical_cv_split_output[["information"]][["parameters"]][["sample_size"]] <- nrow(cleaned_data)
  categorical_cv_split_output[["information"]][["parameters"]][["additional_arguments"]] <- list(...)
  #Store classes
  categorical_cv_split_output[["classes"]][[response_var]] <- names(table(cleaned_data[,response_var]))
  #Create formula string
  categorical_cv_split_output[["formula"]] <- formula <- as.formula(paste(response_var, "~", paste(feature_vec, collapse = " + ")))
  #Get names and create a dictionary to convert to numeric if logistic model is chosen
  if(model_type == "logistic"){
    #Start at 0
    class_position <- 0
    for(class in names(table(cleaned_data[,response_var]))){
      categorical_cv_split_output[["class_dictionary"]][[as.character(class)]] <- class_position 
      class_position <- class_position  + 1
    }
  }
  if(stratified == TRUE){
    #Initialize list; initializing for ordering output purposes
    categorical_cv_split_output[["class_indices"]] <- list()
    #Get proportions
    categorical_cv_split_output[["class_proportions"]] <- table(cleaned_data[,response_var])/sum(table(cleaned_data[,response_var]))
    #Get the indices with the corresponding categories and ass to list
    for(class in names(categorical_cv_split_output[["class_proportions"]])){
      categorical_cv_split_output[["class_indices"]][[class]]  <- which(cleaned_data[,response_var] == class)
    }
  }
  #Stratified sampling
  if(!is.null(split)){
    if(stratified == TRUE){
      #Get out of .stratified_sampling
      stratified.sampling_output <- .stratified_sampling(data = cleaned_data,type = "split", split = split, output = categorical_cv_split_output, response_var = response_var, random_seed = random_seed)
      #Create training and test set
      training_data <- cleaned_data[stratified.sampling_output$output$sample_indices$split$training,]
      test_data <- cleaned_data[stratified.sampling_output$output$sample_indices$split$test,]
      #Extract updated categorical_cv_split_output output list
      categorical_cv_split_output <- stratified.sampling_output$output
    }else{
      #Create test and training set
      training_indices <- sample(1:nrow(cleaned_data),size = round(nrow(cleaned_data)*split,0),replace = F)
      training_data <- cleaned_data[training_indices,]
      test_data <- cleaned_data[-training_indices,]
      #Store indices in list
      categorical_cv_split_output[["sample_indices"]][["split"]] <- list()
      categorical_cv_split_output[["sample_indices"]][["split"]][["training"]] <- c(1:nrow(cleaned_data))[training_indices]
      categorical_cv_split_output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(cleaned_data))[-training_indices]
    }
    #Create data table
    categorical_cv_split_output[["metrics"]][["split"]] <- data.frame(matrix(nrow = 2, ncol = 1))
    colnames(categorical_cv_split_output[["metrics"]][["split"]]) <- "Set"
    categorical_cv_split_output[["metrics"]][["split"]][1:2,"Set"] <- c("Training","Test")
  }
  #Adding information to data frame
  if(!is.null(k)){
    categorical_cv_split_output[["metrics"]][["cv"]] <- data.frame(matrix(nrow = 1,ncol = 1))
    colnames(categorical_cv_split_output[["metrics"]][["cv"]]) <- "Fold"
    #Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    #Initialize list to store fold indices; third subindex needs to be initialized
    categorical_cv_split_output[["sample_indices"]][["cv"]] <- list()
    #Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      #Initialize list to store fold proportions; third level
      categorical_cv_split_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- .stratified_sampling(data = cleaned_data, type = "k-fold", output = categorical_cv_split_output,
                                                        response_var = response_var, k = k,
                                                        random_seed = random_seed)
      #Collect output
      categorical_cv_split_output <- stratified.sampling_output$output
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
        categorical_cv_split_output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
        #Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        #Add indices to list
        categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
    #Reorder list
    metrics_position <- which(names(categorical_cv_split_output) == "metrics")
    categorical_cv_split_output <- c(categorical_cv_split_output[-metrics_position],categorical_cv_split_output[metrics_position])
  }
  #Ensure model type is lowercase
  model_type <- tolower(model_type)
  #Add it plus one to the iterator if k is not null
  iterator <- ifelse(is.null(k), 1, k + 1)
  #Initialize list to store training models
  if(!is.null(split)){
    if(save_models == TRUE){
      categorical_cv_split_output[[paste0(model_type,"_models")]][["split"]] <- list()
    }
    #Create iterator vector
    iterator_vector <- 1:iterator
  }
  if(!is.null(k)){
    if(save_models == TRUE){
      categorical_cv_split_output[[paste0(model_type,"_models")]][["cv"]] <- list()
    }
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
    cleaned_data[,response_var] <- sapply(data[,response_var], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
    training_data[,response_var] <- sapply(training_data[,response_var], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
    test_data[,response_var] <- sapply(test_data[,response_var], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
  }
  #First iteration will always be the evaluation for the traditional data split method
  for(i in iterator_vector){
    if(i == 1){
      #Assigning data split matrices to new variable 
      model_data <- training_data
      #Ensure columns have same levels
      if(model_type == "svm"){
        for(col in names(data_levels)){
          levels(model_data[,col]) <- data_levels[[col]]
        }
      }
    }else{
      #After the first iteration the cv begins, the training set is assigned to a new variable
      model_data <- cleaned_data[-c(categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
      validation_data <- cleaned_data[c(categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
      #Ensure columns have same levels
      if(model_type == "svm"){
        for(col in names(data_levels)){
          levels(model_data[,col]) <- levels(validation_data[,col]) <- data_levels[[col]]
        }
      }
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
             classes <- as.numeric(unlist(categorical_cv_split_output[["class_dictionary"]]))
             class_names <- names(categorical_cv_split_output[["class_dictionary"]])
           },
           class_names <- classes <- categorical_cv_split_output[["classes"]][[response_var]])
    #Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("Training","Test")){
        if(j == "Test"){
          #Assign validation data to new variables
          if(remove_untrained_observation == TRUE){
            model_data <- .remove_untrained_observations(trained_data = model_data, test_data = test_data, response_var = response_var)
          }else{
            model_data <- test_data
          }
        } else{
          if(save_models == TRUE){
            categorical_cv_split_output[[paste0(model_type,"_models")]][["split"]][[tolower(j)]] <- model 
          }
        }
        if(save_data == TRUE){
          #Store dataframe
          categorical_cv_split_output[["data"]][[tolower(j)]] <- model_data
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
        classification_accuracy <- sum(model_data[,response_var] == prediction_vector)/length(model_data[,response_var])
        categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),"Classification Accuracy"] <- classification_accuracy
        #Class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          #Sum of true positives
          true_pos <- sum(model_data[,response_var][which(model_data[,response_var] == class)] == prediction_vector[which(model_data[,response_var] == class)])
          #Sum of false negatives
          false_neg <- sum(model_data[, response_var] == class & prediction_vector != class)
          #Sum of the false positive
          false_pos <- sum(prediction_vector == class) - true_pos
          #Calculate metrics and store in dataframe
          precision <- true_pos/(true_pos + false_pos)
          recall <- true_pos/(true_pos + false_neg)
          f1 <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Precision", class_names[class_position])] <- precision
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Recall", class_names[class_position])] <- recall
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s F1", class_names[class_position])] <- f1
          class_position <- class_position + 1
          #Warning is a metric is NA
          if(any(is.na(c(classification_accuracy,precision,recall,f1)))){
            metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy,precision,recall,f1)))]
            warning(sprintf("at least on metric could not be calculated for class %s - %s set: %s",class,tolower(j),paste(metrics, collapse = ",")))
          }
        }
      }
    } else{
      if(all(!is.null(k),(i-1) <= k)){
        #Assign validation data to new variables
        if(remove_untrained_observation == TRUE){
          model_data <- .remove_untrained_observations(trained_data = model_data, test_data = validation_data, response_var = response_var, fold = i-1)
        }else{
          model_data <- validation_data
        }
        if(save_models == TRUE){
          categorical_cv_split_output[[paste0(model_type,"_models")]][["cv"]][[sprintf("fold %s", i-1)]] <- model
        }
        # Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = model_data)},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = model_data, type = "response")
                 prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = model_data)},
               prediction_vector <- predict(model, newdata = model_data)$class
        )
        if(save_data == TRUE){
          #Store dataframe
          categorical_cv_split_output[["data"]][[sprintf("fold %s",i-1)]] <- model_data
        }
        #Calculate classification accuracy for fold
        classification_accuracy <- sum(model_data[,response_var] == prediction_vector)/length(model_data[,response_var])
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- classification_accuracy
        #Reset class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          #Sum of true positives
          true_pos <- sum(model_data[,response_var][which(model_data[,response_var] == class)] == prediction_vector[which(model_data[,response_var] == class)])
          #Sum of false negatives
          false_neg <- sum(model_data[, response_var] == class & prediction_vector != class)
          #Sum of the false positive
          false_pos <- sum(prediction_vector == class) - true_pos
          #Calculate metrics and store in dataframe
          precision <- true_pos/(true_pos + false_pos)
          recall <- true_pos/(true_pos + false_neg)
          f1 <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
          #Add metrics to dataframe
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Precision", class_names[class_position])] <- precision
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Recall", class_names[class_position])] <- recall
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s F1", class_names[class_position])] <- f1
          #Warning is a metric is NA
          if(any(is.na(c(classification_accuracy,precision,recall,f1)))){
            metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy,precision,recall,f1)))]
            warning(sprintf("at least on metric could not be calculated for class %s - fold %s: %s",class,i-1,paste(metrics, collapse = ",")))
          }
          class_position <- class_position + 1
        }
      }
    }
    #Calculate mean, standard deviation, and standard error for cross validation
    if(all(!is.null(k),(i-1) == k)){
      idx <- nrow(categorical_cv_split_output[["metrics"]][["cv"]] )
      categorical_cv_split_output[["metrics"]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
      #Calculate mean, standard deviation, and sd for each column except for fold
      for(colname in colnames(categorical_cv_split_output[["metrics"]][["cv"]] )[colnames(categorical_cv_split_output[["metrics"]][["cv"]] ) != "Fold"]){
        #Create vector containing corresponding column name values for each fold
        num_vector <- categorical_cv_split_output[["metrics"]][["cv"]][1:idx, colname]
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(k)
      }
    }
  }
  #Make list a vswift class
  class(categorical_cv_split_output) <- "vswift"
  return(categorical_cv_split_output)
}
