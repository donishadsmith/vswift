categorical.cv.split  <- function(data = NULL, y_col = NULL,x_col = NULL,k = NULL, split = 0.8, model_type = NULL, stratified = FALSE, plot_metrics = FALSE, random_seed = NULL){
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
  
  
  # Checking if inputs are valid
  error.handling(data = data, y_col = y_col, x_col = x_col, k = k, split = split, model_type = model_type, stratified = stratified, plot_metrics = plot_metrics, random_seed = random_seed)
  
  #Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  
  # Print parameter information
  cat(sprintf("Model Type: %s\n\n", model_type))
  cat(sprintf("Response Variable: %s\n\n", ifelse(is.character(y_col), y_col, colnames(data)[y_col])))
  if (is.null(x_col)) {
    response_var <- ifelse(is.character(y_col), y_col, colnames(data)[y_col])
    feature_vec <- colnames(data)[colnames(data) != response_var]
  } else {
    feature_vec <- ifelse(all(is.character(x_col)), x_col, colnames(data)[x_col])
  }
  cat(sprintf("Features: %s\n\n", paste(feature_vec, collapse = ", ")))
  cat(sprintf("K: %s\n\n", k))
  cat(sprintf("Split: %s\n\n", split))
  cat(sprintf("Stratified Sampling: %s\n\n", stratified))
  cat(sprintf("Plot Metrics: %s\n", plot_metrics))
  
  # Get response and predictors
  if(is.character(y_col)){
    y_col <- which(colnames(data) == y_col)
  }
  #Remove rows with missing data
  cleaned_data <- data[complete.cases(data),]
  if(is.null(x_col)){
    x <-  cleaned_data[,-y_col]
  } else{
    x <-  cleaned_data[,x_col]
  }
  y <- cleaned_data[,y_col]
  
  # Initialize output list
  categorical.cv.split_output <- list()
  
  # Get category names
  # Convert y to numeric ranging starting with 0, while preserving original names in a dictionary
  categorical.cv.split_output[["class_dict"]] <- list()
  categories_length <- 0
  for(category in names(table(y))){
    if(is.numeric(y)){
      y[which(y == as.numeric(category))] <- categories_length
    } else{
      y[which(y == category)] <- categories_length
    }
    categorical.cv.split_output[["class_dict"]][[as.character(categories_length)]] <- category
    categories_length <- categories_length + 1
  }
  y <- as.numeric(y)
  #Initialize output
  categorical.cv.split_output[["sample_indices"]] <- list()
  
  #Stratified sampling
  if(stratified == TRUE){
    stratified.sampling_output <- stratified.sampling(type = "split", split = split, output = categorical.cv.split_output, y = y, random_seed = random_seed)
    X_train <- as.matrix(x[stratified.sampling_output$training,]); Y_train <- as.matrix(y[stratified.sampling_output$training])
    X_val <- as.matrix(x[stratified.sampling_output$test,]); Y_val <- as.matrix(y[stratified.sampling_output$test])
    categorical.cv.split_output <- stratified.sampling_output$output
    class_indices <- stratified.sampling_output$class_indices
  } else{
    #Data split
    training_indices <- sample(1:nrow(x),size = round(nrow(x)*split,0),replace = F)
    X_train <- as.matrix(x[training_indices,]);  Y_train <- as.matrix(y[training_indices])
    X_val <- as.matrix(x[-training_indices,]); Y_val <- as.matrix(y[-training_indices])
    categorical.cv.split_output[["sample_indices"]][["training"]] <- c(1:nrow(x))[training_indices]
    categorical.cv.split_output[["sample_indices"]][["test"]] <- c(1:nrow(x))[-training_indices]
  }
  #Create data table
  set_metrics <- data.frame(matrix(nrow = 2, ncol = 1))
  colnames(set_metrics) <- "Set"
  set_metrics[1:2,"Set"] <- c("Training","Test")
  #Adding information to data frame
  if(!(is.null(k))){
    k_metrics <- data.frame(matrix(nrow = 1,ncol = 1))
    colnames(k_metrics) <- "Fold"
    #Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(x))
    # Initialize list to store fold indices
    categorical.cv.split_output[["sample_indices"]][["cv"]] <- list()
    
    #Creating non-overlapping folds while adding rownames to matrix
    
    if(stratified == TRUE){
      categorical.cv.split_output[["class_proportions"]] <- table(y)/sum(table(y))
      names(categorical.cv.split_output[["class_proportions"]]) <- as.vector(categorical.cv.split_output[["class_dict"]])
      # Initialize list to store fold proportions
      categorical.cv.split_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- stratified.sampling(type = "k-fold", output = categorical.cv.split_output, y = y,
                                               k_metrics = k_metrics, k = k,
                                               class_indices = class_indices,
                                               random_seed = random_seed)
      # Collect output
      categorical.cv.split_output <- stratified.sampling_output$output
      k_metrics <- stratified.sampling_output$k_metrics
    } else{
      #Get floor
      fold_size_vector <- rep(floor(nrow(x)/k),k)
      excess <- nrow(x) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:k,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      # random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:k){
        #Add name to dataframe
        k_metrics[i,"Fold"] <- sprintf("Fold %s",i)
        #Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        #Add indices to list
        categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
  }
  #Assign 1 or k to iterator
  iterator <- ifelse(is.null(k),1,k)
  #Ensure model type is lowercase
  model_type <- tolower(model_type)
  #Add it plus one to the iterator if k is not null
  iterator <- ifelse(!(is.null(k)), iterator + 1, iterator)
  # Initialize list to store training models
  categorical.cv.split_output[[sprintf("%s models", model_type)]] <- list()
  
  if(!(is.null(k))){
    categorical.cv.split_output[[sprintf("%s models", model_type)]][["cv"]] <- list()
  }
  # Convert variables to characters so that models will predict the original variable
  if(model_type != "logistic"){
    y <- sapply(y, function(x) categorical.cv.split_output[["class_dict"]][[as.character(x)]])
    Y_train <- sapply(Y_train, function(x) categorical.cv.split_output[["class_dict"]][[as.character(x)]])
    Y_val <- sapply(Y_val, function(x) categorical.cv.split_output[["class_dict"]][[as.character(x)]])
  }
  
  #First iteration will always be the evaluation for the traditional data split method
  for(i in 1:iterator){
    if(i == 1){
      # Assigning data split matrices to new variable 
      X_data <- X_train
      Y_data <- Y_train
    } else{
      # After the first iteration the cv begins, the training set is assigned to a new variable
      X_data <- as.matrix(x[-c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]) ; Y_data <- as.matrix(y[-c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]])])
      X_val <- as.matrix(x[c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]),]) ; Y_val <- as.matrix(y[c(categorical.cv.split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]])])
    }
    #Generate model depending on chosen model_type
    
    switch(model_type,
           "lda" = {model <- MASS::lda(as.factor(Y_data) ~ X_data)},
           "qda" = {model <- MASS::qda(as.factor(Y_data) ~ X_data)},
           "logistic" = {model <- glm(Y_data ~ X_data, family = "binomial")},
           "svm" = {model <- e1071::svm(as.factor(Y_data) ~ X_data)},
           "naivebayes" = {model <- naivebayes::naive_bayes(x = X_data, y= as.factor(Y_data))}
    )
    # Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("training","test")){
        
        if(j == "test"){
          #Assign test set to new variable
          X_data <- X_val
          Y_data <- Y_val
        } else{
          categorical.cv.split_output[[sprintf("%s models", model_type)]][["split"]][[j]] <- model
        }
        # Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = data.frame(X_data))},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = data.frame(X_data), type = "response")
                 prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = data.frame(X_data))},
               prediction_vector <- predict(model, newx = X_data)$class
        )
        set_metrics[which(set_metrics$Set == j),"Classification Accuracy"] <- sum(Y_data == prediction_vector)/length(Y_data)
        # Plot metrics for training and test
        if(all(j == "Test", plot_metrics == TRUE)){
          plot(x = 1:2, y = set_metrics[1:2,"Classification Accuracy"] , ylim = c(0,1), xlab = "Set", ylab = "Classification Accuracy", xaxt = "n")
          axis(1, at = 1:2, labels = c("Training","Test"))
        }
        
        
        for(category in names(categorical.cv.split_output[["class_dict"]])){
          if(model_type == "logistic"){
            #Sum of category that was correctly guessed
            true_pos <- sum(Y_data[which(Y_data == as.numeric(category))] == prediction_vector[which(Y_data  == as.numeric(category))])
            #Sum of the missed category
            false_neg <- abs(true_pos  - length(Y_data[which(Y_data  == as.numeric(category))]))
            #Sum of the classified category; Taking "One Class vs. Rest" approach
            false_pos <- length(which(prediction_vector[-which(Y_data  == as.numeric(category))] == as.numeric(category)))
          } else{
            converted_category <- categorical.cv.split_output[["class_dict"]][[category]]
            #Sum of category that was correctly guessed
            true_pos <- sum(Y_data[which(Y_data == converted_category)] == prediction_vector[which(Y_data  == converted_category)])
            #Sum of the missed category
            false_neg <- abs(true_pos  - length(Y_data[which(Y_data  == converted_category)]))
            #Sum of the classified category; Taking "One Class vs. Rest" approach
            false_pos <- length(which(prediction_vector[-which(Y_data  == converted_category)] == converted_category))
          }
          set_metrics[which(set_metrics$Set == j),sprintf("Category: %s Precision", categorical.cv.split_output[["class_dict"]][[category]])] <- true_pos/(true_pos + false_pos)
          set_metrics[which(set_metrics$Set == j),sprintf("Category: %s Recall", categorical.cv.split_output[["class_dict"]][[category]])] <- true_pos/(true_pos + false_neg)
          set_metrics[which(set_metrics$Set == j),sprintf("Category: %s F1", categorical.cv.split_output[["class_dict"]][[category]])] <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
          # Plot metrics for training and test
          
          if(all(j == "Test", plot_metrics == TRUE)){
            plot(x = 1:2, y = set_metrics[1:2,sprintf("Category: %s Precision", categorical.cv.split_output[["class_dict"]][[category]])] , ylim = c(0,1), xlab = "Set", ylab = "Precision" , xaxt = "n",
                 main = paste("Category:",categorical.cv.split_output[["class_dict"]][[category]]))
            axis(1, at = 1:2, labels = c("Training","Test"))
            
            plot(x = 1:2, y = set_metrics[1:2,sprintf("Category: %s Recall", categorical.cv.split_output[["class_dict"]][[category]])] , ylim = c(0,1), xlab = "Set", ylab = "Recall" , xaxt = "n",
                 main = paste("Category:",categorical.cv.split_output[["class_dict"]][[category]]))
            axis(1, at = 1:2, labels = c("Training","Test"))
            
            plot(x = 1:2, y = set_metrics[1:2,sprintf("Category: %s F1", categorical.cv.split_output[["class_dict"]][[category]])] , ylim = c(0,1), xlab = "Set", ylab = "F1" , xaxt = "n",
                 main = paste("Category:",categorical.cv.split_output[["class_dict"]][[category]]))
            axis(1, at = 1:2, labels = c("Training","Test"))
            
          }
        }
        
      }
    } else{
      if(all(!(is.null(k)),(i-1) <= k)){
        #Assign validation set to new variables
        X_data <- X_val
        Y_data <- Y_val
        categorical.cv.split_output[[sprintf("%s models", model_type)]][["cv"]][[sprintf("fold %s", i-1)]] <- model
        # Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = data.frame(X_data))},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = data.frame(X_data), type = "response")
                 prediction_vector <- ifelse(prediction_vector > 0.5, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = data.frame(X_data))},
               prediction_vector <- predict(model, newx = X_data)$class
        )
        # Calculate classification accuracy for fold
        k_metrics[which(k_metrics$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- sum(Y_data == prediction_vector)/nrow(Y_data)
        for(category in names(categorical.cv.split_output[["class_dict"]])){
          if(model_type == "logistic"){
            #Sum of category that was correctly guessed
            true_pos <- sum(Y_data[which(Y_data == as.numeric(category))] == prediction_vector[which(Y_data  == as.numeric(category))])
            #Sum of the missed category
            false_neg <- abs(true_pos  - length(Y_data[which(Y_data  == as.numeric(category))]))
            #Sum of the classified category; Taking "One Class vs. Rest" approach
            false_pos <- length(which(prediction_vector[-which(Y_data  == as.numeric(category))] == as.numeric(category)))
          } else{
            converted_category <- categorical.cv.split_output[["class_dict"]][[category]]
            #Sum of category that was correctly guessed
            true_pos <- sum(Y_data[which(Y_data == converted_category)] == prediction_vector[which(Y_data  == converted_category)])
            #Sum of the missed category
            false_neg <- abs(true_pos  - length(Y_data[which(Y_data  == converted_category)]))
            #Sum of the classified category; Taking "One Class vs. Rest" approach
            false_pos <- length(which(prediction_vector[-which(Y_data  == converted_category)] == converted_category))
          }
          
          k_metrics[which(k_metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Category: %s Precision", categorical.cv.split_output[["class_dict"]][[category]])] <- true_pos/(true_pos + false_pos)
          k_metrics[which(k_metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Category: %s Recall", categorical.cv.split_output[["class_dict"]][[category]])] <- true_pos/(true_pos + false_neg)
          k_metrics[which(k_metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Category: %s F1", categorical.cv.split_output[["class_dict"]][[category]])] <- 2/(1/(true_pos/(true_pos + false_pos)) + 1/(true_pos/(true_pos + false_neg)))
        }
        #Calculate final metrics and plot
      }
      if(all(!(is.null(k)),(i-1) == k)){
        # To get the correct category for plot title
        category_idx <- 1
        #Get the last row index
        idx <- nrow(k_metrics)
        #Initialize new metrics
        k_metrics[(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
        # Calculate mean, standard deviation, and sd for each column except for fold
        for(colname in colnames(k_metrics)[colnames(k_metrics) != "Fold"]){
          # Create vector containing corresponding column name values for each fold
          num_vector <- k_metrics[1:idx, colname]
          k_metrics[which(k_metrics$Fold == "Mean CV:"),colname] <- mean(num_vector)
          k_metrics[which(k_metrics$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
          k_metrics[which(k_metrics$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(k)
          #Split column name
          split_vector <- unlist(strsplit(colname, split = " "))
          #Plot metrics
          if(plot_metrics == TRUE){
            # depending on column name, plotting is handled slightly differently
            if("Classification" %in% split_vector){
              plot(x = 1:k, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
              axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
            } else{
              # Get correct metric name for plot y title
              y_name <- c("Precision","Recall","F1")[which(c("Precision","Recall","F1") %in% split_vector)]
              
              plot(x = 1:k, y = num_vector, ylim = c(0,1), xlab = "K-folds", ylab = y_name, main = paste("Category: ",categorical.cv.split_output[["class_dict"]][[category_idx]]), xaxt = "n")
              axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
              # Add 1 to `category_idx` when `y_name == "Recall"` to get correct category plot title
              if(y_name == "F1"){
                category_idx <- category_idx + 1
              }
            }
            # Add mean and standard deviation to the plot
            abline(h = mean(num_vector), col = "red", lwd = 1)
            abline(h = mean(num_vector) + sd(num_vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
            abline(h = mean(num_vector) - sd(num_vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
          }
        }
      }
    }
  }
  # Create list of output depending on if k validation is done or not
  categorical.cv.split_output[["metrics"]] <- list()
  categorical.cv.split_output[["metrics"]][["split"]] <- set_metrics
  
  if(!(is.null(k))){
    categorical.cv.split_output[["metrics"]][["cv"]] <- k_metrics
  } 

  return(categorical.cv.split_output)
}
