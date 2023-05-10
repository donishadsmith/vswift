#' Perform Train-Test Split and/or K-Fold Cross-Validation with optional stratified sampling for classification data
#'
#' `classCV` performs a train-test split and/or k-fold cross validation
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
#'                   "lda" (Linear Discriminant Analysis), "qda" (Quadratic Discriminant Analysis), 
#'                   "logistic" (Logistic Regression), "svm" (Support Vector Machines), "naivebayes" (Naive Bayes), 
#'                   "ann" (Artificial Neural Network), "knn" (K-Nearest Neighbors), "decisiontree" (Decision Tree), 
#'                   "randomforest" (Random Forest), "multinom" (Multinomial Logistic Regression), "gbm" (Gradiant Boosted Modeling).
#'                   For "knn", the optimal k will be used unless specified with `ks =`.
#'                   For "ann", `size =` must be specified as an additional argument.
#' @param threshold  A number from 0.3 to 0.7 indicating representing the decision boundary for logistic regression.                 
#' @param stratified A logical value indicating if stratified sampling should be used. Default = FALSE.
#' @param random_seed A numerical value for the random seed. Default = NULL.
#' @param impute_method A character indicating the imputation method to use. Available options: "simple" (simple imputation) and "missforest" (Random Forest Imputation from missForest package). For simple imputation, the specific method used on columns with missing quantitative data depends on its distribution. A shapiro test is conducted to assess
#' the normality of the data. If the shapiro test is significant (p-value < 0.05), then missing data is replaced with the column median if it is not significant
#' (p-value >= 0.5), then the missing values are replaced with the column mean. Missing qualitative data is replaced with the column mode.
#' Default == False. If data is missing and `impute_method == NULL`, observations with missing data will be removed. Default == NULL.
#' @param impute_args A list containing additional arguments to pass to missForest for Random Forest Imputation. Available options: "maxiter","ntree","variablewise","decreasing","verbose",
#' "mtry", "replace", "classwt", "cutoff","strata", "sampsize", "nodesize", "maxnodes". Note: For specific information about each parameter, please refer to the missForest documentation. Default = NULL.
#' @param remove_obs A logical value to remove observations with categorical predictors from the test/validation set
#'                   that have not been observed during model training. Some algorithms may produce an error if this occurs. Default = FALSE.
#' @param save_models A logical value to save models during train-test splitting and/or k-fold cross validation. Default = FALSE.
#' @param save_data A logical value to save all training and test/validation sets during train-test splitting and/or k-fold cross validation. Default = FALSE.
#' @param final_model A logical value to use all complete observations in the input data for model training. Default = FALSE.
#' @param ... Additional arguments specific to the chosen classification algorithm.
#'            Please refer to the corresponding algorithm's documentation for additional arguments and their descriptions.
#' 
#' @section Model-specific additional arguments:
#'   Each model type accepts additional arguments specific to the classification algorithm. The available arguments for each model type are:
#'
#'   - "lda": grouping, prior, method, nu
#'   - "qda": grouping, prior, method, nu
#'   - "logistic": weights, start, etastart, mustart, offset, control, contrasts, intercept, singular.ok, type, maxit
#'   - "svm": scale, type, kernel, degree, gamma, coef0, cost, nu, class.weights, cachesize, tolerance, epsilon, shrinking, cross, probability, fitted
#'   - "naivebayes": prior, laplace, usekernel, usepoisson
#'   - "ann": weights, size, Wts, mask, linout, entropy, softmax, censored, skip, rang, decay, maxit, Hess, trace, MaxNWts, abstol, reltol
#'   - "knn": kmax, ks, distance, kernel, scale, contrasts, ykernel
#'   - "decisiontree": weights, method, parms, control, cost
#'   - "randomforest": ntree, mtry, weights, replace, classwt, cutoff, strata, nodesize, maxnodes, importance, localImp, nPerm, proximity, oob.prox, norm.votes, do.trace, keep.forest, corr.bias, keep.inbag
#'   - "multinom": weights, Hess
#'   - "gbm": distribution, weights, var.monotone, n.trees, interaction.depth,n.minobsinnode, shrinkage, train.faction, cv.folds, keep.data, verbose, class.stratify.cv, n.cores
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
#'   - "multinom": multinom() from nnet package
#'   - "gbm": xgb.train() from xgboost package
#'                    
#' @return A list containing the results of train-test splitting and/or k-fold cross-validation,
#'         including imputation information (if specified), performance metrics, information on the class distribution in the training, test sets, and folds (if applicable), 
#'         saved models (if specified), and saved datasets (if specified), and a final model (if specified).
#' 
#' @seealso \code{\link{print.vswift}}, \code{\link{plot.vswift}}
#' 
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#' result <- classCV(data = iris, target = "Species", split = 0.8, model_type = "lda")
#' 
#' # Print parameters and metrics
#' print(result)
#' 
#' # Plot metrics
#' plot(result)
#'
#' # Perform 5-fold cross-validation using Gradient Boosted Model
#' result <- classCV(data = iris, target = "Species", n_folds = 5, model_type = "gbm",params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10)
#' 
#' # Print parameters and metrics
#' print(result)
#' 
#' # Plot metrics
#' plot(result)
#' 
#' @export
classCV <- function(data = NULL, target = NULL, predictors = NULL, split = NULL, n_folds = NULL, model_type = NULL, threshold = 0.5, stratified = FALSE, random_seed = NULL, impute_method = NULL, impute_args = NULL, remove_obs = FALSE, save_models = FALSE, save_data = FALSE, final_model = FALSE,...){
  # Ensure model type is lowercase
  model_type <- tolower(model_type)
  # Checking if inputs are valid
  vswift:::.error_handling(data = data, target = target, predictors = predictors, n_folds = n_folds, split = split, model_type = model_type, threshold = threshold, stratified = stratified, random_seed = random_seed, impute_method = impute_method, impute_args = impute_args, call = "classCV", ...)
  # Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  # Creating response variable
  target <- ifelse(is.character(target), target, colnames(data)[target])
  # Creating feature vector
  if(is.null(predictors)){
    predictor_vec <- colnames(data)[colnames(data) != target]
  } else {
    if(all(is.character(predictors))){
      predictor_vec <- predictors
    } else {
      predictor_vec <- colnames(data)[predictors]
    }
  }
  # Ensure data is factor
  data[,target] <- factor(data[,target])
  # Perform simple imputation
  if(!is.null(impute_method)){
    # Get data with missing columns
    missing_columns <- unique(data.frame(which(is.na(data), arr.ind = T))$col)
    if(length(missing_columns) == 0){
      preprocessed_data <- data
      warning("no missing data detected")
    } else {
      impute_output <- vswift:::.impute(data = data, missing_columns = missing_columns,
                                        impute_method = impute_method, impute_args = impute_args)
      preprocessed_data <- impute_output[["preprocessed_data"]]
      
      # If error occurs remove missing data that has not been replaced
      preprocessed_data <- preprocessed_data[complete.cases(preprocessed_data ),]
    }
  } else {
    preprocessed_data <- data[complete.cases(data),]
  }
  if(model_type == "svm"){
    # Turn to factor and get levels
    data_levels <- list()
    if(class(preprocessed_data[,target]) == "factor"){
      data_levels[[target]] <- levels(preprocessed_data[,target])
    } else {
      # Turn response column to character
      preprocessed_data[,target] <- as.character(preprocessed_data[,target])
    }
    # Get character columns
    character_columns <- as.vector(sapply(preprocessed_data,function(x) is.character(x)))
    factor_columns <- as.vector(sapply(preprocessed_data,function(x) is.factor(x)))
    # Get column names
    columns <- colnames(preprocessed_data)[character_columns]
    columns <- c(columns, colnames(preprocessed_data)[factor_columns])
    if(!length(columns) == 0){
      for(col in columns){
        if(is.character(preprocessed_data[,col])){
          preprocessed_data[,col] <- factor(preprocessed_data[,col])
        }
        data_levels[[col]] <- levels(preprocessed_data[,col])
      }
    }
  }
  # Initialize output list
  classCV_output <- list()
  classCV_output[["information"]][["analysis_type"]] <- "classification"
  classCV_output[["information"]][["parameters"]] <- list()
  classCV_output[["information"]][["parameters"]][["predictors"]] <- predictor_vec
  classCV_output[["information"]][["parameters"]][["target"]]  <- target
  classCV_output[["information"]][["parameters"]][["model_type"]] <- model_type
  if(model_type == "logistic"){
    classCV_output[["information"]][["parameters"]][["threshold"]] <- threshold
  }
  classCV_output[["information"]][["parameters"]][["n_folds"]]  <- n_folds
  classCV_output[["information"]][["parameters"]][["stratified"]]  <- stratified
  classCV_output[["information"]][["parameters"]][["split"]]  <- split
  classCV_output[["information"]][["parameters"]][["random_seed"]]  <- random_seed
  classCV_output[["information"]][["parameters"]][["missing_data"]]  <- nrow(data) - nrow(preprocessed_data)
  classCV_output[["information"]][["parameters"]][["impute_method"]] <- impute_method
  if(!is.null(impute_method)){
    if(impute_method == "missforest"){
      classCV_output[["information"]][["parameters"]][["impute_args"]] <- impute_args
    }
  }
  if(exists("impute_output")) classCV_output[["information"]][["imputation_info"]] <- impute_output[["impute_info"]]
  if(classCV_output[["information"]][["parameters"]][["missing_data"]] > 0){
    warning(sprintf("dataset contains %s observations with incomplete data only complete observations will be used"
                    , classCV_output[["information"]][["parameters"]][["missing_data"]] ))
  }
  classCV_output[["information"]][["parameters"]][["sample_size"]] <- nrow(preprocessed_data)
  classCV_output[["information"]][["parameters"]][["additional_arguments"]] <- list(...)
  # Store classes
  classCV_output[["classes"]][[target]] <- names(table(factor(preprocessed_data[,target])))
  # Create formula string
  classCV_output[["information"]][["formula"]] <- formula <- as.formula(paste(target, "~", paste(predictor_vec, collapse = " + ")))
  # conversion for later
  conversion_needed <- FALSE
  # Get names and create a dictionary to convert to numeric if logistic model is chosen
  if(model_type %in% c("logistic", "gbm")){
    counter <- 0
    new_classes <- c()
    for(class in names(table(preprocessed_data[,target]))){
      new_classes <- c(new_classes, paste(class, "=", counter, collapse = " "))
      classCV_output[["class_dictionary"]][[class]] <- counter
      counter <- counter + 1
    }
    conversion_needed <- TRUE
    warning(sprintf("form logistic regression and gradient boosted modeling classes must be numerics; classes are now encoded: %s", paste(new_classes, collapse = ", ")))
  }
  if(stratified == TRUE){
    # Initialize list; initializing for ordering output purposes
    classCV_output[["class_indices"]] <- list()
    # Get proportions
    classCV_output[["class_proportions"]] <- table(preprocessed_data[,target])/sum(table(preprocessed_data[,target]))
    # Get the indices with the corresponding categories and ass to list
    for(class in names(classCV_output[["class_proportions"]])){
      classCV_output[["class_indices"]][[class]]  <- which(preprocessed_data[,target] == class)
    }
  }
  # Stratified sampling
  if(!is.null(split)){
    if(stratified == TRUE){
      # Get out of .stratified_sampling
      stratified.sampling_output <- vswift:::.stratified_sampling(data = preprocessed_data,type = "split", split = split, output = classCV_output, target = target, random_seed = random_seed)
      # Create training and test set
      training_data <- preprocessed_data[stratified.sampling_output$output$sample_indices$split$training,]
      test_data <- preprocessed_data[stratified.sampling_output$output$sample_indices$split$test,]
      # Extract updated classCV_output output list
      classCV_output <- stratified.sampling_output$output
    } else {
      # Create test and training set
      training_indices <- sample(1:nrow(preprocessed_data),size = round(nrow(preprocessed_data)*split,0),replace = F)
      training_data <- preprocessed_data[training_indices,]
      test_data <- preprocessed_data[-training_indices,]
      # Store indices in list
      classCV_output[["sample_indices"]][["split"]] <- list()
      classCV_output[["sample_indices"]][["split"]][["training"]] <- c(1:nrow(preprocessed_data))[training_indices]
      classCV_output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(preprocessed_data))[-training_indices]
    }
    # Create data table
    classCV_output[["metrics"]][["split"]] <- data.frame("Set" = c("Training","Test"))
  }
  # Adding information to data frame
  if(!is.null(n_folds)){
    classCV_output[["metrics"]][["cv"]] <- data.frame("Fold" = NA)
    # Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    # Initialize list to store fold indices; third subindex needs to be initialized
    classCV_output[["sample_indices"]][["cv"]] <- list()
    # Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      # Initialize list to store fold proportions; third level
      classCV_output[["sample_proportions"]][["cv"]] <- list()
      stratified.sampling_output <- vswift:::.stratified_sampling(data = preprocessed_data, type = "k-fold", output = classCV_output, k = n_folds,
                                                        target = target,random_seed = random_seed)
      # Collect output
      classCV_output <- stratified.sampling_output$output
    } else {
      # Get floor
      fold_size_vector <- rep(floor(nrow(preprocessed_data)/n_folds),n_folds)
      excess <- nrow(preprocessed_data) - sum(fold_size_vector)
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
        classCV_output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
        # Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        # Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        # Add indices to list
        classCV_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
    # Reorder list
    metrics_position <- which(names(classCV_output) == "metrics")
    classCV_output <- c(classCV_output[-metrics_position],classCV_output[metrics_position])
  }
  # Add it plus one to the iterator if n_folds is not null
  iterator <- ifelse(is.null(n_folds), 1, n_folds + 1)
  if(save_data == TRUE) classCV_output[["data"]][["preprocessed_data"]] <- preprocessed_data
  # Initialize list to store training models
  if(!is.null(split)){
    if(save_models == TRUE) classCV_output[[paste0(model_type,"_models")]][["split"]] <- list()
    # Create iterator vector
    iterator_vector <- 1:iterator
  }
  if(!is.null(n_folds)){
    if(save_models == TRUE) classCV_output[[paste0(model_type,"_models")]][["cv"]] <- list()
    # Create iterator vector
    if(!is.null(split)){
      #Create iterator vector
      iterator_vector <- 1:iterator
    } else {
      # Create iterator vector
      iterator_vector <- 2:iterator
    }
  }
  # Convert variables to characters so that models will predict the original variable
  if(conversion_needed == TRUE){
    preprocessed_data[,target] <- sapply(preprocessed_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
    training_data[,target] <- sapply(training_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
    test_data[,target] <- sapply(test_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
  }
  
  # First iteration will always be the evaluation for the traditional data split method
  if(any(!is.null(split), !is.null(n_folds))){
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
      } else {
        # After the first iteration the cv begins, the training set is assigned to a new variable
        model_data <- preprocessed_data[-c(classCV_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
        validation_data <- preprocessed_data[c(classCV_output[["sample_indices"]][["cv"]][[sprintf("fold %s",(i-1))]]), ]
        # Ensure columns have same levels
        if(model_type == "svm"){
          for(col in names(data_levels)){
            levels(model_data[,col]) <- levels(validation_data[,col]) <- data_levels[[col]]
          }
        }
      }
      # Suitable for gbm
      # Generate model depending on chosen model_type
      model <- vswift:::.generate_model(model_type = model_type, formula = formula, predictors = predictors, target = target, model_data = model_data, ...)
      # Create variables used in for loops to calculate precision, recall, and f1
      switch(model_type,
             "logistic" = {
               classes <- as.numeric(unlist(classCV_output[["class_dictionary"]]))
               class_names <- names(classCV_output[["class_dictionary"]])
             },
             "gbm" = {
               classes <- as.numeric(unlist(classCV_output[["class_dictionary"]]))
               class_names <- names(classCV_output[["class_dictionary"]])
             },
             class_names <- classes <- classCV_output[["classes"]][[target]])
      # Perform classification accuracy for training and test data split
      if(i == 1){
        for(j in c("Training","Test")){
          if(j == "Test"){
            # Assign validation data to new variables
            if(remove_obs == TRUE){
              model_data <- vswift:::.remove_obs(training_data = model_data, test_data = test_data, target = target)
            } else {
              model_data <- test_data
            }
          } else {
            if(save_models == TRUE) classCV_output[[paste0(model_type,"_models")]][["split"]][[tolower(j)]] <- model 
          }
          # Store dataframe is save_data is TRUE
          if(save_data == TRUE) classCV_output[["data"]][[tolower(j)]] <- model_data
          # Get prediction
          prediction_vector <- vswift:::.prediction(model_type = model_type, model = model, model_data = model_data, predictors = predictors, target = target, class_dict = classCV_output[["class_dictionary"]])
          # Calculate classification accuracy
          classification_accuracy <- sum(model_data[,target] == prediction_vector)/length(model_data[,target])
          classCV_output[["metrics"]][["split"]][which(classCV_output[["metrics"]][["split"]]$Set == j),"Classification Accuracy"] <- classification_accuracy
          # Class positions to get the name of the class in class_names
          class_position <- 1
          for(class in classes){
            metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector, model_data = model_data)
            # Add information to dataframes
            classCV_output[["metrics"]][["split"]][which(classCV_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
            classCV_output[["metrics"]][["split"]][which(classCV_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
            classCV_output[["metrics"]][["split"]][which(classCV_output[["metrics"]][["split"]]$Set == j),sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
            class_position <- class_position + 1
            # Warning is a metric is NA
            if(any(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))){
              metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))]
              warning(sprintf("at least on metric could not be calculated for class %s - fold %s: %s",class,i-1,paste(metrics, collapse = ",")))
            }
          }
        }
      } else {
        if(all(!is.null(n_folds),(i-1) <= n_folds)){
          # Assign validation data to new variables
          if(remove_obs == TRUE){
            model_data <- vswift:::.remove_obs(training_data = model_data, test_data = validation_data, target = target, fold = i-1)
          } else {
            model_data <- validation_data
          }
          if(save_models == TRUE) classCV_output[[paste0(model_type,"_models")]][["cv"]][[sprintf("fold %s", i-1)]] <- model
          # Get prediction
          prediction_vector <- vswift:::.prediction(model_type = model_type, model = model, model_data = model_data, predictors = predictors, target = target, class_dict = classCV_output[["class_dictionary"]])
          # Save data
          if(save_data == TRUE) classCV_output[["data"]][[sprintf("fold %s",i-1)]] <- model_data
          # Calculate classification accuracy for fold
          classification_accuracy <- sum(model_data[,target] == prediction_vector)/length(model_data[,target])
          classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- classification_accuracy
          # Reset class positions to get the name of the class in class_names
          class_position <- 1
          for(class in classes){
            # Calculate metrics
            metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector, model_data = model_data)
            # Add metrics to dataframe
            classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
            classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
            classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == sprintf("Fold %s",i-1)), sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
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
        idx <- nrow(classCV_output[["metrics"]][["cv"]] )
        classCV_output[["metrics"]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
        # Calculate mean, standard deviation, and sd for each column except for fold
        for(colname in colnames(classCV_output[["metrics"]][["cv"]] )[colnames(classCV_output[["metrics"]][["cv"]] ) != "Fold"]){
          # Create vector containing corresponding column name values for each fold
          num_vector <- classCV_output[["metrics"]][["cv"]][1:idx, colname]
          classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
          classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
          classCV_output[["metrics"]][["cv"]][which(classCV_output[["metrics"]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(n_folds)
        }
      }
    }
  }
  # Generate final model
  if(final_model == TRUE){
    # Generate model depending on chosen model_type
    classCV_output[["final_model"]]  <- vswift:::.generate_model(model_type = model_type, formula = formula, predictors = predictors, target = target, model_data = model_data, ...)
  }
  # Make list a vswift class
  class(classCV_output) <- "vswift"
  return(classCV_output)
}
