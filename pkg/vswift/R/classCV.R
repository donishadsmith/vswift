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
#' @param model_type A character string or list indicating the classification algorithm to use. Available options:
#'                   "lda" (Linear Discriminant Analysis), "qda" (Quadratic Discriminant Analysis), 
#'                   "logistic" (Logistic Regression), "svm" (Support Vector Machines), "naivebayes" (Naive Bayes), 
#'                   "ann" (Artificial Neural Network), "knn" (K-Nearest Neighbors), "decisiontree" (Decision Tree), 
#'                   "randomforest" (Random Forest), "multinom" (Multinomial Logistic Regression), "gbm" (Gradient Boosting Machine).
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
#' @param mod_args  list of named sub-lists. Each sub-list corresponds to a model specified in the `model_type` parameter, and contains the parameters to be passed 
#' to the respective model. Default = NULL.
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
#'   - "lda": prior, method, nu
#'   - "qda": prior, method, nu
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
#' # Perform 5-fold cross-validation a train-test split with an 80% training set using multiple models
#' 
#' result <- classCV(data = iris, target = 5, split = 0.8, model_type = c("decisiontree","gbm","knn", "ann","svm"), 
#' n_folds = 3, mod_args = list("knn" = list(ks = 3), "ann" = list(size = 10), "gbm" = list(params = list(objective = "multi:softprob",num_class = 3,eta = 0.3,max_depth = 6), nrounds = 10)), 
#' save_data = T, save_models = T, remove_obs = T, stratified = T)
#' 
#' # Print parameters and metrics
#' print(result)
#' 
#' # Plot metrics
#' plot(result)
#' @author Donisha Smith
#' 
#' 
#' @export
classCV <- function(data, target, predictors = NULL, split = NULL, n_folds = NULL, model_type, threshold = 0.5, stratified = FALSE, random_seed = NULL, impute_method = NULL, impute_args = NULL,
                    mod_args = NULL, remove_obs = FALSE, save_models = FALSE, save_data = FALSE, final_model = FALSE,...){
  
  # Ensure model type is lowercase
  if(!is.null(model_type)) model_type <- tolower(model_type)
  
  # Checking if inputs are valid
  vswift:::.error_handling(data = data, target = target, predictors = predictors, n_folds = n_folds, split = split, model_type = model_type, threshold = threshold, stratified = stratified, random_seed = random_seed, impute_method = impute_method, impute_args = impute_args, mod_args = mod_args, call = "classCV", ...)
  
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
  create_factor_output <- vswift:::.create_factor(data = data, target = target, model_type = model_type)
  data <- create_factor_output[["data"]] 
  if("svm" %in% model_type){
    data_levels <- create_factor_output[["data_levels"]]
  }
  
  # Perform imputation
  imputation_output <- vswift:::.imputation(data = data, impute_method = impute_method, impute_args = impute_args)
  preprocessed_data <- imputation_output[["preprocessed_data"]]
 
  # Store information
  classCV_output <- vswift:::.store_parameters(data = data, preprocessed_data = preprocessed_data, predictor_vec = predictor_vec, target = target, model_type = model_type,
                               threshold = threshold, split = split, n_folds = n_folds, stratified = stratified, random_seed = random_seed, impute_method = impute_method,
                               impute_args = impute_args, imputation_output = imputation_output, mod_args = mod_args, ...)
  
  # Get formula
  formula <- classCV_output[["formula"]]
  
  # Create class dictionary
  if(any(model_type %in% c("logistic", "gbm"))){
    classCV_output <- vswift:::.create_dictionary(preprocessed_data = preprocessed_data, target = target, classCV_output = classCV_output)
  }
  
  # Initialize empty vector for iteration
  iterator_vector <- c()
  
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
      # Extract updated classCV_output output list
      classCV_output <- stratified.sampling_output$output
    } else {
      # Create test and training set
      training_indices <- sample(1:nrow(preprocessed_data),size = round(nrow(preprocessed_data)*split,0),replace = F)
      # Store indices in list
      classCV_output[["sample_indices"]][["split"]] <- list()
      classCV_output[["sample_indices"]][["split"]][["training"]] <- c(1:nrow(preprocessed_data))[training_indices]
      classCV_output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(preprocessed_data))[-training_indices]
    }
    # Assignments to iterator vector
    iterator_vector <- c("Training","Test")
  }
  # Adding information to data frame
  if(!is.null(n_folds)){
    # Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(data))
    # Initialize list to store fold indices; third subindex needs to be initialized
    classCV_output[["sample_indices"]][["cv"]] <- list()
    # Creating non-overlapping folds while adding rownames to matrix
    if(stratified == TRUE){
      # Initialize list to store fold proportions; third level
      classCV_output[["sample_proportions"]][["cv"]] <- list()
      stratified_sampling_output <- vswift:::.stratified_sampling(data = preprocessed_data, type = "k-fold", output = classCV_output, k = n_folds,
                                                        target = target, random_seed = random_seed)
      # Collect output
      classCV_output <- stratified_sampling_output$output
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
      # Random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:n_folds){
        # Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        # Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        # Add indices to list
        classCV_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
      }
    }
    iterator_vector <- c(iterator_vector, paste("Fold", 1:n_folds))
  }
  
  # Save data
  if(save_data == TRUE) classCV_output[["saved_data"]][["preprocessed_data"]] <- preprocessed_data
 
  
  # Move logistic and gbm to back of list
  if(all(length(model_type) > 1, any(c("logistic","gbm") %in% model_type))){
    x <- which(model_type %in% c("logistic","gbm"))
    model_type <- c(model_type[-x],model_type[x])
  }
  
  # Variable to keep track of logistic and gbm
  model_eval_tracker <- 1
  # Perform model evaluation
  for(model_name in model_type){
    # Convert target variable
    if(model_name %in% c("logistic","gbm")){
      if(model_eval_tracker == 1){
      preprocessed_data[,target] <- sapply(preprocessed_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
      }
      model_eval_tracker <- model_eval_tracker + 1
    }
    
    # Check if split or cv specified then perform evaluation
    if(any(!is.null(split), !is.null(n_folds))){
      for(i in iterator_vector){
        #Create split word
        split_word <- unlist(strsplit(i, split = " "))
        if("Training" %in% split_word){
          # Assigning data split matrices to new variable 
          model_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["training"]], ]
          validation_data <- preprocessed_data[classCV_output[["sample_indices"]][["split"]][["test"]], ]
        } else if("Fold" %in% split_word){
          model_data <- preprocessed_data[-c(classCV_output[["sample_indices"]][["cv"]][[tolower(i)]]), ]
          validation_data <- preprocessed_data[c(classCV_output[["sample_indices"]][["cv"]][[tolower(i)]]), ]
        }
        
        # Ensure columns have same levels
        if(model_name == "svm"){
          for(col in names(data_levels)){
            levels(model_data[,col]) <- levels(validation_data[,col]) <- data_levels[[col]]
          }
        }
        
        # Generate model depending on chosen model_type
        model <- vswift:::.generate_model(model_type = model_name, formula = formula, predictors = predictors, target = target, model_data = model_data, mod_args = mod_args, ...)
        
        # Variable to select correct dataframe
        method <- ifelse("Fold" %in% split_word, "cv", "split")
        col <- ifelse(method == "cv", "Fold", "Set")

        # Create dataframe for validation testing
        if(any("Test" %in% split_word, "Fold" %in% split_word) & remove_obs == TRUE){
          remove_obs_output <- vswift:::.remove_obs(training_data = model_data, test_data = validation_data, target = target, iter = i, method = method, preprocessed_data = preprocessed_data,
                                                    output = classCV_output, stratified = stratified)
          prediction_data <- remove_obs_output[["test_data"]]
          classCV_output <- remove_obs_output[["output"]]
        } else {
          if("Training" %in% split_word){
            prediction_data <- model_data
          } else {
            prediction_data <- validation_data
          }
        }
        
        # Create variables used in for loops to calculate precision, recall, and f1
        if(model_name %in% c("logistic","gbm")){
          classes <- as.numeric(unlist(classCV_output[["class_dictionary"]]))
          class_names <- names(classCV_output[["class_dictionary"]])
        } else {
          class_names <- classes <- classCV_output[["classes"]][[target]]
        }
        
        # Save data
        if(save_models == TRUE) classCV_output[["saved_data"]][[method]][[tolower(i)]] <- preprocessed_data[classCV_output[["sample_indices"]][[method]][[tolower(i)]],]
        # Save model
        if(save_models == TRUE) classCV_output[["saved_models"]][[model_name]][[method]] <- model
        
        # Create dataframe
        if(i == "Training"){
          classCV_output[["metrics"]][[model_name]][[method]] <- data.frame("Set" = c("Training", "Test"))
        } else if(i == "Fold 1"){
          classCV_output[["metrics"]][[model_name]][[method]] <- data.frame("Fold" = paste("Fold",1:n_folds))
        }
        
        # Get prediction
        prediction_vector <- vswift:::.prediction(model_type = model_name, model = model, prediction_data = prediction_data, predictors = predictors, target = target, threshold = threshold, class_dict = classCV_output[["class_dictionary"]])
        
        # Calculate classification accuracy
        classification_accuracy <- sum(prediction_data[,target] == prediction_vector)/length(prediction_data[,target])
        classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == i),"Classification Accuracy"] <- classification_accuracy
        # Class positions to get the name of the class in class_names
        class_position <- 1
        for(class in classes){
          metrics_list <- vswift:::.calculate_metrics(class = class, target = target, prediction_vector = prediction_vector, prediction_data = prediction_data)
          # Add information to dataframes
          classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == i),sprintf("Class: %s Precision", class_names[class_position])] <- metrics_list[["precision"]]
          classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == i),sprintf("Class: %s Recall", class_names[class_position])] <- metrics_list[["recall"]]
          classCV_output[["metrics"]][[model_name]][[method]][which(classCV_output[["metrics"]][[model_name]][[method]][,col] == i),sprintf("Class: %s F-Score", class_names[class_position])] <- metrics_list[["f1"]]
          class_position <- class_position + 1
          # Warning is a metric is NA
          if(any(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))){
            metrics <- c("classification accuracy","precision","recall","f-score")[which(is.na(c(classification_accuracy, metrics_list[["precision"]], metrics_list[["recall"]], metrics_list[["f1"]])))]
            warning(sprintf("at least on metric could not be calculated for class %s - %s: %s",class,i,paste(metrics, collapse = ",")))
          }
        }
        
        # Reset class position
        class_position <- 1
        
        # Calculate mean, standard deviation, and standard error for cross validation
        if(i == paste("Fold", n_folds)){
          idx <- nrow(classCV_output[["metrics"]][[model_name]][["cv"]])
          classCV_output[["metrics"]][[model_name]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
          # Calculate mean, standard deviation, and sd for each column except for fold
          for(colname in colnames(classCV_output[["metrics"]][[model_name]][["cv"]] )[colnames(classCV_output[["metrics"]][[model_name]][["cv"]] ) != "Fold"]){
            # Create vector containing corresponding column name values for each fold
            num_vector <- classCV_output[["metrics"]][[model_name]][["cv"]][1:idx, colname]
            classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector)
            classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector)
            classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector)/sqrt(n_folds)
          }
        }
      }
    }
    
    # Generate final model
    if(final_model == TRUE){
      # Generate model depending on chosen model_type
      classCV_output[["final_model"]][[model_name]]  <- vswift:::.generate_model(model_type = model_name, formula = formula, predictors = predictors, target = target, model_data = preprocessed_data, mod_args = mod_args, ...)
    } 
  }

  # Make list a vswift class
  class(classCV_output) <- "vswift"
  return(classCV_output)
}


