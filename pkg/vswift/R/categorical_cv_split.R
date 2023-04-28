#' Perform Train-Test Split and/or K-Fold Cross-Validation with optional stratified sampling for classification data
#'
#' `categorical_cv_split` performs a train-test split and/or k-fold cross validation
#' on classification data using various classification algorithms.
#'
#'
#' @param data A data frame containing the dataset.
#' @param target The target variable's numerical index or name in the data frame.
#' @param predictors A vector of numerical indices or names for the predictors in the data frame.
#'              If not specified, all variables except the response variable will be used as predictors.
#' @param split A number from 0.5 and 0.9 for the proportion of data to use for the training set,
#'              leaving the rest for the test set. If not specified, train-test splitting will not be done.
#' @param n_folds An integer from 3 and 30 for the number of folds to use. If not specified,
#'               k-fold cross validation will not be performed.             
#' @param model_type A character string indicating the classification algorithm to use. Available options:
#'                   "lda", "qda", "logistic", "svm", "naivebayes", "ann", "knn", "decisiontree", "randomforest".
#'                   For "knn", the optimal k will be used unless specified with `ks =`.
#'                   For "ann", `size =` must be specified as an additional argument.
#' @param threshold  A number from 0.3 to 0.7 indicating representing the decision boundary for logistic regression.                 
#' @param stratified A logical value indicating if stratified sampling should be used. Default = FALSE.
#' @param random_seed A numerical value for the random seed. Default is NULL.
#' @param remove_obs A logical value to remove observations with categorical predictors from the test/validation set
#'                   that have not been observed during model training. Some algorithms may produce an error if this occurs. Default is FALSE.
#' @param save_models A logical value to save models during train-test splitting and/or k-fold cross validation. Default is FALSE.
#' @param save_data A logical value to save all training and test/validation sets during train-test splitting and/or k-fold cross validation. Default is FALSE.
#' @param ... Additional arguments specific to the chosen classification algorithm.
#'            Please refer to the corresponding algorithm's documentation for additional arguments and their descriptions.
#' 
#' @section Model-specific additional arguments:
#'   Each model type accepts additional arguments specific to the classification algorithm. The available arguments for each model type are:
#'
#'   - "lda": grouping, prior, method, nu
#'   - "qda": grouping, prior, method, nu
#'   - "logistic": weights, start, etastart, mustart, offset, control, contrasts, intercept, singular.ok, type
#'   - "svm": scale, type, kernel, degree, gamma, coef0, cost, nu, class.weights, cachesize, tolerance, epsilon, shrinking, cross, probability, fitted
#'   - "naivebayes": prior, laplace, usekernel, usepoisson
#'   - "ann": weights, size, Wts, mask, linout, entropy, softmax, censored, skip, rang, decay, maxit, Hess, trace, MaxNWts, abstol, reltol
#'   - "knn": kmax, ks, distance, kernel, scale, contrasts, ykernel
#'   - "decisiontree": weights, method, parms, control, cost
#'   - "randomforest": ntree, mtry, weights, replace, classwt, cutoff, strata, nodesize, maxnodes, importance, localImp, nPerm, proximity, oob.prox, norm.votes, do.trace, keep.forest, corr.bias, keep.inbag
#' 
#' @section Functions used from packages for each model type:
#'
#'   - "lda": lda() from MASS package
#'   - "qda": qda() from MASS package
#'   - "logistic": glm() from base package with family = "binomial"
#'   - "svm": svm() from e1071 package
#'   - "naivebayes": naive_bayes() from naivebayes package
#'   - "ann": nnet() from nnet package
#'   - "knn": train.kknn() from kknn package
#'   - "decisiontree": rpart() from rpart package
#'   - "randomforest": randomForest() from randomForest package 
#'                    
#' @return A list containing the results of train-test splitting and/or k-fold cross-validation,
#'         including performance metrics, information on the class distribution in the training, test sets, and folds (if applicable), 
#'         saved models (if specified), and saved datasets (if specified).
#' 
#' @seealso \code{\link{print.vswift}}, \code{\link{plot.vswift}}
#' 
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#' result <- categorical_cv_split(iris, target = "Species", split = 0.8, model_type = "lda")
#'
#' # Perform 5-fold cross-validation using QDA
#' result <- categorical_cv_split(iris, target = "Species", n_folds = 5, model_type = "qda")
#' @export
categorical_cv_split <- function(data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = 0.5, stratified = FALSE, random_seed = NULL, remove_obs = FALSE, save_models = FALSE, save_data = FALSE,...){
  # Ensure model type is lowercase
  model_type <- tolower(model_type)
  # Checking if inputs are valid
  vswift:::.error_handling(data = data, target = target, predictors = predictors, n_folds = n_folds, split = split, model_type = model_type, threshold = threshold, stratified = stratified, random_seed = random_seed, call = "categorical_cv_split")
  # Check if additional arguments are valid
  if(length(list(...)) > 0){
    vswift:::.check_additional_arguments(model_type = model_type, ...)
  }
  # Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  # Creating response variable
  target <- ifelse(is.character(target), target, colnames(data)[target])
  # Creating feature vector
  if (is.null(predictors)){
    predictor_vec <- colnames(data)[colnames(data) != target]
  }else {
    if(all(is.character(predictors))){
      predictor_vec <- predictors
    }else{
      predictor_vec <- colnames(data)[predictors]
    }
  }
  # Remove rows with missing data
  cleaned_data <- data[complete.cases(data),]
  if(model_type == "svm"){
    # Turn to factor and get levels
    data_levels <- list()
    if(class(cleaned_data[,target]) == "factor"){
      data_levels[[target]] <- levels(cleaned_data[,target])
    }else{
      # Turn response column to character
      cleaned_data[,target] <- as.character(cleaned_data[,target])
    }
    # Get character columns
    character_columns <- sapply(cleaned_data,function(x) is.character(x))
    factor_columns <- sapply(cleaned_data,function(x) is.factor(x))
    columns <- c(character_columns,factor_columns) 
    # Get character column names
    columns <- colnames(cleaned_data)[columns]
    for(col in columns){
      if(is.character(cleaned_data[,col])){
        cleaned_data[,col] <- factor(cleaned_data[,col])
      }
      data_levels[[col]] <- levels(cleaned_data[,col])
    }
  }
  if(!model_type %in% c("svm","logistic")){
    cleaned_data[,target] <- factor(cleaned_data[,target])
  }
  # Initialize output list
  categorical_cv_split_output <- list()
  categorical_cv_split_output[["information"]][["analysis_type"]] <- "classification"
  categorical_cv_split_output[["information"]][["parameters"]] <- list()
  categorical_cv_split_output[["information"]][["parameters"]][["predictors"]] <- predictor_vec
  categorical_cv_split_output[["information"]][["parameters"]][["target"]]  <- target
  categorical_cv_split_output[["information"]][["parameters"]][["model_type"]] <- model_type
  if(model_type == "logistic"){
    categorical_cv_split_output[["information"]][["parameters"]][["threshold"]] <- threshold
  }
  categorical_cv_split_output[["information"]][["parameters"]][["n_folds"]]  <- n_folds
  categorical_cv_split_output[["information"]][["parameters"]][["stratified"]]  <- stratified
  categorical_cv_split_output[["information"]][["parameters"]][["split"]]  <- split
  categorical_cv_split_output[["information"]][["parameters"]][["random_seed"]]  <- random_seed
  categorical_cv_split_output[["information"]][["parameters"]][["missing_data"]]  <- nrow(data) - nrow(cleaned_data)
  categorical_cv_split_output[["information"]][["parameters"]][["sample_size"]] <- nrow(cleaned_data)
  categorical_cv_split_output[["information"]][["parameters"]][["additional_arguments"]] <- list(...)
  # Store classes
  categorical_cv_split_output[["classes"]][[target]] <- names(table(factor(cleaned_data[,target])))
  # Create formula string
  categorical_cv_split_output[["information"]][["formula"]] <- formula <- as.formula(paste(target, "~", paste(predictor_vec, collapse = " + ")))
  # Get names and create a dictionary to convert to numeric if logistic model is chosen
  if(model_type == "logistic"){
    # Check if target is in appropriate format for logistic
    if(any(!all(cleaned_data[,target] %in% c(0,1)), !is.numeric(cleaned_data[,target]))){
      # Convert to numeric is a character vector of 0's and 1's
      if(all(cleaned_data[,target] %in% c("0","1"))){
        cleaned_data[,target] <- as.numeric(cleaned_data[,target])
        categorical_cv_split_output[["class_dictionary"]][["0"]] <- 0
        categorical_cv_split_output[["class_dictionary"]][["1"]] <- 1
      }else if(any(cleaned_data[,target] %in% c("0","1"))){
        # Handle case if one class is "0" or "1"
        factor_class <- unique(cleaned_data[which(!cleaned_data[,target]%in% c("0","1")), target])
        if("0" %in% cleaned_data[,target]){
          categorical_cv_split_output[["class_dictionary"]][["0"]] <- 0
          categorical_cv_split_output[["class_dictionary"]][[factor_class]] <- 1
        }else{
          categorical_cv_split_output[["class_dictionary"]][[factor_class]] <- 0
          categorical_cv_split_output[["class_dictionary"]][["1"]] <- 1
        }
        warning(sprintf("for logistic regression target variable must be vector of consisting of 0's and 1's; classes are now encoded: %s", paste(factor_class,
                                                                                                                                                  "=",
                                                                                                                                                  categorical_cv_split_output[["class_dictionary"]][[factor_class]],
                                                                                                                                                  collapse = " ")))
      }else{
        # Handle case to convert both classes
        cleaned_data[,target] <- factor(cleaned_data[,target])
        # Start at 0
        class_position <- 0
        new_classes <- c()
        for(class in names(table(cleaned_data[,target]))){
          categorical_cv_split_output[["class_dictionary"]][[as.character(class)]] <- class_position 
          new_classes <- c(new_classes, paste(class, "=", class_position, collapse = " "))
          class_position <- class_position  + 1
          }
        warning(sprintf("for logistic regression target variable must be vector of consisting of 0's and 1's; classes are now encoded: %s", paste(new_classes, collapse = ", ")))
      }
    }else{
      categorical_cv_split_output[["class_dictionary"]][["0"]] <- 0
      categorical_cv_split_output[["class_dictionary"]][["1"]] <- 1
      }
  }
  if(stratified == TRUE){
    # Initialize list; initializing for ordering output purposes
    categorical_cv_split_output[["class_indices"]] <- list()
    # Get proportions
    categorical_cv_split_output[["class_proportions"]] <- table(cleaned_data[,target])/sum(table(cleaned_data[,target]))
    # Get the indices with the corresponding categories and ass to list
    for(class in names(categorical_cv_split_output[["class_proportions"]])){
      categorical_cv_split_output[["class_indices"]][[class]]  <- which(cleaned_data[,target] == class)
    }
  }
  # Stratified sampling
  if(!is.null(split)){
    if(stratified == TRUE){
      # Get out of .stratified_sampling
      stratified.sampling_output <- vswift:::.stratified_sampling(data = cleaned_data,type = "split", split = split, output = categorical_cv_split_output, target = target, random_seed = random_seed)
      # Create training and test set
      training_data <- cleaned_data[stratified.sampling_output$output$sample_indices$split$training,]
      test_data <- cleaned_data[stratified.sampling_output$output$sample_indices$split$test,]
      # Extract updated categorical_cv_split_output output list
      categorical_cv_split_output <- stratified.sampling_output$output
    }else{
      # Create test and training set
      training_indices <- sample(1:nrow(cleaned_data),size = round(nrow(cleaned_data)*split,0),replace = F)
      training_data <- cleaned_data[training_indices,]
      test_data <- cleaned_data[-training_indices,]
      # Store indices in list
      categorical_cv_split_output[["sample_indices"]][["split"]] <- list()
      categorical_cv_split_output[["sample_indices"]][["split"]][["training"]] <- c(1:nrow(cleaned_data))[training_indices]
      categorical_cv_split_output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(cleaned_data))[-training_indices]
    }
    # Create data table
    categorical_cv_split_output[["metrics"]][["split"]] <- data.frame("Set" = c("Training","Test"))
  }
  # Adding information to data frame
  if(!is.null(n_folds)){
    categorical_cv_split_output[["metrics"]][["cv"]] <- data.frame("Fold" = NA)
    # Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    # Initialize list to store fold indices; third subindex needs to be initialized
    categorical_cv_split_output[["sample_indices"]][["cv"]] <- list()
    # Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      # Initialize list to store fold proportions; third level
      categorical_cv_split_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- vswift:::.stratified_sampling(data = cleaned_data, type = "k-fold", output = categorical_cv_split_output, k = n_folds,
                                                        target = target,random_seed = random_seed)
      # Collect output
      categorical_cv_split_output <- stratified.sampling_output$output
    }else{
      # Get floor
      fold_size_vector <- rep(floor(nrow(cleaned_data)/n_folds),n_folds)
      excess <- nrow(cleaned_data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:n_folds,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      # random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:n_folds){
        # Add name to dataframe
        categorical_cv_split_output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
        # Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        # Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        # Add indices to list
        categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
    # Reorder list
    metrics_position <- which(names(categorical_cv_split_output) == "metrics")
    categorical_cv_split_output <- c(categorical_cv_split_output[-metrics_position],categorical_cv_split_output[metrics_position])
  }
  # Add it plus one to the iterator if n_folds is not null
  iterator <- ifelse(is.null(n_folds), 1, n_folds + 1)
  # Initialize list to store training models
  if(!is.null(split)){
    if(save_models == TRUE){
      categorical_cv_split_output[[paste0(model_type,"_models")]][["split"]] <- list()
    } 
    # Create iterator vector
    iterator_vector <- 1:iterator
  }
  if(!is.null(n_folds)){
    if(save_models == TRUE){
      categorical_cv_split_output[[paste0(model_type,"_models")]][["cv"]] <- list()
    }
    # Create iterator vector
    if(!is.null(split)){
      #Create iterator vector
      iterator_vector <- 1:iterator
    } else{
      # Create iterator vector
      iterator_vector <- 2:iterator
    }
  }
  
  # Convert variables to characters so that models will predict the original variable
  if(all(model_type == "logistic", !all(cleaned_data[,target] %in% c(0,1)))){
    cleaned_data[,target] <- sapply(cleaned_data[,target], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
    training_data[,target] <- sapply(training_data[,target], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
    test_data[,target] <- sapply(test_data[,target], function(x) categorical_cv_split_output[["class_dictionary"]][[as.character(x)]])
  }
  
  # First iteration will always be the evaluation for the traditional data split method
  for(i in iterator_vector){
    if(i == 1){
      # Assigning data split matrices to new variable 
      model_data <- training_data
      # Ensure columns have same levels is svm model chosen. The svm may encounter an error is this is not done
      if(model_type == "svm"){
        for(col in names(data_levels)){
          levels(model_data[,col]) <- data_levels[[col]]
        }
      }
    }else{
      # After the first iteration the cv begins, the training set is assigned to a new variable
      model_data <- cleaned_data[-c(categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
      validation_data <- cleaned_data[c(categorical_cv_split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
      # Ensure columns have same levels
      if(model_type == "svm"){
        for(col in names(data_levels)){
          levels(model_data[,col]) <- levels(validation_data[,col]) <- data_levels[[col]]
        }
      }
    }
    
    # Generate model depending on chosen model_type
    switch(model_type,
           # Use double colon to avoid cluttering user space
           "lda" = {model <- MASS::lda(formula, data = model_data, ...)},
           "qda" = {model <- MASS::qda(formula, data = model_data, ...)},
           "logistic" = {model <- glm(formula, data = model_data , family = "binomial", ...)},
           "svm" = {model <- e1071::svm(formula, data = model_data, ...)},
           "naivebayes" = {model <- naivebayes::naive_bayes(formula = formula, data = model_data, ...)},
           "ann" = {model <- nnet::nnet(formula = formula, data = model_data, ...)},
           "knn" = {model <- kknn::train.kknn(formula = formula, data = model_data, ...)},
           "decisiontree" = {model <- rpart::rpart(formula = formula, data = model_data, ...)},
           "randomforest" = {model <- randomForest::randomForest(formula = formula, data = model_data, ...)}
    )
    
    # Create variables used in for loops to calculate precision, recall, and f1
    switch(model_type,
           "logistic" = {
             classes <- as.numeric(unlist(categorical_cv_split_output[["class_dictionary"]]))
             class_names <- names(categorical_cv_split_output[["class_dictionary"]])
           },
           class_names <- classes <- categorical_cv_split_output[["classes"]][[target]])
    # Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("Training","Test")){
        if(j == "Test"){
          # Assign validation data to new variables
          if(remove_obs == TRUE){
            model_data <- vswift:::.remove_obs(training_data = model_data, test_data = test_data, target = target)
          }else{
            model_data <- test_data
          }
        }else{
          if(save_models == TRUE){
            categorical_cv_split_output[[paste0(model_type,"_models")]][["split"]][[tolower(j)]] <- model 
          }
        }
        # Store dataframe is save_data is TRUE
        if(save_data == TRUE){
          # Store dataframe
          categorical_cv_split_output[["data"]][[tolower(j)]] <- model_data
        }
        # Get prediction
        switch(model_type,
               "svm" = {prediction_vector <- predict(model, newdata = model_data)},
               "logistic" = {
                 prediction_vector <- predict(model, newdata  = model_data, type = "response")
                 prediction_vector <- ifelse(prediction_vector > threshold, 1, 0)},
               "naivebayes" = {prediction_vector <- predict(model, newdata = model_data)},
               "ann" = {prediction_vector <- predict(model, newdata = model_data, type = "class")},
               "knn" = {prediction_vector <- predict(model, newdata = model_data)},
               "decisiontree" = {
                 prediction_df <- predict(model, newdata = model_data)
                 prediction_vector <- c()
                 #Iterate over dataframe and select colname with the highest probability
                 for(row in 1:nrow(prediction_df)){
                   prediction_vector <- c(prediction_vector,colnames(prediction_df)[which.max(prediction_df[row,])])}},
               "randomforest" = {prediction_vector <- predict(model, newdata = model_data)},
               prediction_vector <- predict(model, newdata = model_data)$class
        )
        # Calculate classification accuracy
        classification_accuracy <- sum(model_data[,target] == prediction_vector)/length(model_data[,target])
        categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),"Classification Accuracy"] <- classification_accuracy
        # Class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector, model_data = model_data)
          # Add information to dataframes
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
          categorical_cv_split_output[["metrics"]][["split"]][which(categorical_cv_split_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
          class_position <- class_position + 1
          # Warning is a metric is NA
          if(any(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))){
            metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))]
            warning(sprintf("at least on metric could not be calculated for class %s - fold %s: %s",class,i-1,paste(metrics, collapse = ",")))
          }
        }
      }
    } else{
      if(all(!is.null(n_folds),(i-1) <= n_folds)){
        # Assign validation data to new variables
        if(remove_obs == TRUE){
          model_data <- vswift:::.remove_obs(training_data = model_data, test_data = validation_data, target = target, fold = i-1)
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
               "ann" = {prediction_vector <- predict(model, newdata = model_data, type = "class")},
               "knn" = {prediction_vector <- predict(model, newdata = model_data)},
               "decisiontree" = {
                 prediction_df <- predict(model, newdata = model_data)
                 prediction_vector <- c()
                 for(row in 1:nrow(prediction_df)){
                   prediction_vector <- c(prediction_vector,colnames(prediction_df)[which.max(prediction_df[row,])])}},
               "randomforest" = {prediction_vector <- predict(model, newdata = model_data)},
               prediction_vector <- predict(model, newdata = model_data)$class
        )
        if(save_data == TRUE){
          # Store dataframe
          categorical_cv_split_output[["data"]][[sprintf("fold %s",i-1)]] <- model_data
        }
        # Calculate classification accuracy for fold
        classification_accuracy <- sum(model_data[,target] == prediction_vector)/length(model_data[,target])
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- classification_accuracy
        # Reset class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          # Calculate metrics
          metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector, model_data = model_data)
          # Add metrics to dataframe
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
          categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
          # Warning is a metric is NA
          if(any(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))){
            metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))]
            warning(sprintf("at least on metric could not be calculated for class %s - fold %s: %s",class,i-1,paste(metrics, collapse = ",")))
          }
          class_position <- class_position + 1
        }
      }
    }
    # Calculate mean, standard deviation, and standard error for cross validation
    if(all(!is.null(n_folds),(i-1) == n_folds)){
      idx <- nrow(categorical_cv_split_output[["metrics"]][["cv"]] )
      categorical_cv_split_output[["metrics"]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
      # Calculate mean, standard deviation, and sd for each column except for fold
      for(colname in colnames(categorical_cv_split_output[["metrics"]][["cv"]] )[colnames(categorical_cv_split_output[["metrics"]][["cv"]] ) != "Fold"]){
        # Create vector containing corresponding column name values for each fold
        num_vector <- categorical_cv_split_output[["metrics"]][["cv"]][1:idx, colname]
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(n_folds)
      }
    }
  }
  # Make list a vswift class
  class(categorical_cv_split_output) <- "vswift"
  return(categorical_cv_split_output)
}
