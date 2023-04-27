#' categorical_cv_split
#' 
#' categorical_cv_split is used to perform a train-test split and/or k-fold cross validation on classification data
#' 
#' 
#' @param data A data frame.
#' @param y_col The numerical index or name for the response variable in the data frame.
#' @param x_col A vector of numerical indices or names for the features to be used in the data frame. If not specified, all variables in the data frame except for the response variable will be used as features.
#' @param fold_n A numerical value from 3 to 30 indicating the number of folds to use. If not specified, k-fold cross validation will not be performed.
#' @param split A numerical value from 0.5 to 0.9 indicating the proportion of data to use for the training set, leaving the rest for the test set. If not specified, train-test splitting will not be done.
#' @param model_type A character indicating the type of classification algorithm to use. Options: "lda" (Linear Discriminant Analysis),"qda" (Quadratic Discriminant Analysis),"logistic" (Logistic Regression)
#' ,"svm" (Support Vector Machine),"naivebayes" (Naive Bayes),"ann" (Artificial Neural Network),"knn" (K-Nearest Neighbors),"decisiontree" (Decision Tree),"randomforest" (Random Forest).
#' Note that for "knn", the optimal k will be used unless `ks = ` is used as an additional argument and for "ann" `size = ` must be used as an additional argument.
#' @param stratified A logical value specifying if stratified sampling should be used.
#' @param random_seed A numerical value for the random seed to be used. Default is set to NULL.
#' @param save_models A logical value to save the models used for training during train-test splitting and/or k-fold cross validation. Default is set to FALSE. 
#' @param save_data A logical value to save all training and test/validation sets used for during train-test splitting and or k-fold cross validation. Default is set to FALSE. 
#' @param remove_obs A logical value to remove observations with categorical features from the test/validation set that have not been observed during model training. 
#' Note some algorithms may produce an error if this occurs. Default set to FALSE.
#' @param ... Additional arguments specific to the chosen classification algorithm.
#' 
#'   - For "lda" (lda from MASS), default settings are used, but you can modify the following arguments:
#'     - grouping
#'     - prior
#'     - method 
#'     - nu
#'   - For "qda" (qda from MASS), default settings are used, but you can modify the following arguments:
#'     - grouping
#'     - prior
#'     - method
#'     - nu
#'   - For "logistic" (glm from base), default settings are used, with exception of `family = "binomial"`, but you can modify the following arguments: 
#'     - weights
#'     - starts
#'     - etastart
#'     - mustart
#'     - offset
#'     - control
#'     - contrasts
#'     - intercept
#'     - singular.ok
#'     - type
#'   - For "svm" (svm from e1071), default settings are used, but you can modify the following arguments: 
#'     - scale
#'     - type
#'     - kernel
#'     - degree
#'     - gamma
#'     - coef0
#'     - cost
#'     - nu
#'     - class.weights
#'     - cachesize
#'     - tolerance
#'     - epsilon
#'     - shrinking
#'     - cross
#'     - probability
#'     - fitted
#'   - For "naivebayes" (naivebayes from naive_bayes), default settings are used, but you can modify the following arguments:
#'     - prior
#'     - laplace
#'     - usekernel
#'     - usepoisson
#'   - For "ann" (nnet from nnet), default settings are used, but you can modify the following arguments: 
#'     - weights
#'     - size
#'     - Wts
#'     - mask
#'     - linout
#'     - entropy
#'     - softmax
#'     - skip
#'     - rang
#'     - decay
#'     - maxit
#'     - Hess
#'     - trace
#'     - MaxNWts
#'     - abstol
#'     - reltol
#'   - For "knn" (train.kknn from kknn), default settings are used, but you can modify the following arguments: 
#'     - kmax
#'     - ks
#'     - distance
#'     - kernel
#'     - ykernel
#'     - scale
#'     - contrasts
#'   - For "decisiontree" (rpart from rpart), default settings are used, but you can modify the following arguments: 
#'     - weights
#'     - method
#'     - parms
#'     - control
#'     - cost 
#'   - For "randomforest" (randomForest from randomForest), default settings are used, but you can modify the following arguments: 
#'     - ntree
#'     - mtry
#'     - weights
#'     - replace
#'     - classwt
#'     - cutoff
#'     - strata
#'     - nodesize
#'     - maxnodes
#'     - importance
#'     - localImp
#'     - nPerm
#'     - proximity
#'     - oob.prox
#'     - norm.votes
#'     - do.trace
#'     - keep.forest
#'     - corr.bias
#'     - keep.inbag
#' 
#' @return An object of class vswift
#' 
#' @examples
#' 
#' data(iris)
#' 
#' ## Use all predictors with k-nearest neighbors and specify the number of neighbors to use with additional argument `ks = 5`
#' knn_mod <- categorical_cv_split(data = data, y_col = "Species", split = 0.8, fold_n = 5
#' , model_type = "knn", stratified = TRUE, random_seed = 123, ks = 5)
#' 
#' ## Use some predictors with artificial neural network and specificy additional argument `size = 3`
#' ann_mod <- categorical_cv_split(data = data, y_col = "Species", x_col = 1:3, split = 0.8, fold_n = 5
#' , model_type = "ann", stratified = TRUE, random_seed = 123, size = 3)
#' 
#' print(knn_mod)
#' 
#' print(ann_mod)
#' 
#' @export
categorical_cv_split <- function(data = NULL, y_col = NULL, x_col = NULL, split = NULL, fold_n = NULL, model_type = NULL, stratified = FALSE, random_seed = NULL, remove_obs = FALSE, save_models = FALSE, save_data = FALSE,...){
  #Ensure model type is lowercase
  model_type <- tolower(model_type)
  #Checking if inputs are valid
  vswift:::.error_handling(data = data, y_col = y_col, x_col = x_col, fold_n = fold_n, split = split, model_type = model_type, stratified = stratified, random_seed = random_seed, call = "categorical_cv_split")
  #Check if additional arguments are valid
  if(length(list(...)) > 0){
    vswift:::.check_additional_arguments(model_type = model_type, ...)
  }
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
  categorical_cv_split_output[["information"]][["parameters"]][["fold_n"]]  <- fold_n
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
      stratified.sampling_output <- vswift:::.stratified_sampling(data = cleaned_data,type = "split", split = split, output = categorical_cv_split_output, response_var = response_var, random_seed = random_seed)
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
    categorical_cv_split_output[["metrics"]][["split"]] <- data.frame("Set" = c("Training","Test"))
  }
  #Adding information to data frame
  if(!is.null(fold_n)){
    categorical_cv_split_output[["metrics"]][["cv"]] <- data.frame("Fold" = NA)
    #Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    #Initialize list to store fold indices; third subindex needs to be initialized
    categorical_cv_split_output[["sample_indices"]][["cv"]] <- list()
    #Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      #Initialize list to store fold proportions; third level
      categorical_cv_split_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- vswift:::.stratified_sampling(data = cleaned_data, type = "k-fold", output = categorical_cv_split_output, k = fold_n,
                                                        response_var = response_var,random_seed = random_seed)
      #Collect output
      categorical_cv_split_output <- stratified.sampling_output$output
    }else{
      #Get floor
      fold_size_vector <- rep(floor(nrow(cleaned_data)/fold_n),fold_n)
      excess <- nrow(cleaned_data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:fold_n,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      #random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:fold_n){
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
  #Add it plus one to the iterator if fold_n is not null
  iterator <- ifelse(is.null(fold_n), 1, fold_n + 1)
  #Initialize list to store training models
  if(!is.null(split)){
    if(save_models == TRUE){
      categorical_cv_split_output[[paste0(model_type,"_models")]][["split"]] <- list()
    }
    #Create iterator vector
    iterator_vector <- 1:iterator
  }
  if(!is.null(fold_n)){
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
           #Use double colon to avoid cluttering user space
           "lda" = {model <- MASS::lda(formula, data = model_data,...)},
           "qda" = {model <- MASS::qda(formula, data = model_data,...)},
           "logistic" = {model <- glm(formula, data = model_data , family = "binomial",...)},
           "svm" = {model <- e1071::svm(formula, data = model_data,...)},
           "naivebayes" = {model <- naivebayes::naive_bayes(formula = formula, data = model_data,...)},
           "ann" = {model <- nnet::nnet(formula = formula, data = model_data,...)},
           "knn" = {model <- kknn::train.kknn(formula = formula, data = model_data,...)},
           "decisiontree" = {model <- rpart::rpart(formula = formula, data = model_data,...)},
           "randomforest" = {model <- randomForest::randomForest(formula = formula, data = model_data,...)}
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
          if(remove_obs == TRUE){
            model_data <- vswift:::.remove_obs(trained_data = model_data, test_data = test_data, response_var = response_var)
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
          #Sum true negative
          true_neg <- sum(model_data[, response_var] != class & prediction_vector != class)
          #Calculate metrics and store in dataframe
          precision <- true_pos/(true_pos + false_pos)
          recall <- true_pos/(true_pos + false_neg)
          f1 <- 2*(precision*recall)/(precision+recall)
          #Add information to dataframes
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
      if(all(!is.null(fold_n),(i-1) <= fold_n)){
        #Assign validation data to new variables
        if(remove_obs == TRUE){
          model_data <- vswift:::.remove_obs(trained_data = model_data, test_data = validation_data, response_var = response_var, fold = i-1)
        }else{
          model_data <- validation_data
        }
        if(save_models == TRUE){
          categorical_cv_split_output[[paste0(model_type,"_models")]][["cv"]][[sprintf("fold %s", i-1)]] <- model
        }
        #Get prediction
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
          f1 <- 2*(precision*recall)/(precision+recall)
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
    if(all(!is.null(fold_n),(i-1) == fold_n)){
      idx <- nrow(categorical_cv_split_output[["metrics"]][["cv"]] )
      categorical_cv_split_output[["metrics"]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
      #Calculate mean, standard deviation, and sd for each column except for fold
      for(colname in colnames(categorical_cv_split_output[["metrics"]][["cv"]] )[colnames(categorical_cv_split_output[["metrics"]][["cv"]] ) != "Fold"]){
        #Create vector containing corresponding column name values for each fold
        num_vector <- categorical_cv_split_output[["metrics"]][["cv"]][1:idx, colname]
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
        categorical_cv_split_output[["metrics"]][["cv"]][which(categorical_cv_split_output[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(fold_n)
      }
    }
  }
  #Make list a vswift class
  class(categorical_cv_split_output) <- "vswift"
  return(categorical_cv_split_output)
}
