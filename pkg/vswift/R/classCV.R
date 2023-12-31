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
#' @param impute_method A character indicating the imputation method to use. Options include "bag_impute" (Bagged Trees Imputation) and "knn_impute" (KNN Imputation).
#' @param impute_args A list specifying an additional argument for the imputation method. For "bag_impute", the additional argument is "trees" and for "knn_impute", the additional argument is "neighbors". For specific information about each parameter, please refer to the recipes documentation. Default = NULL.
#' @param mod_args  list of named sub-lists. Each sub-list corresponds to a model specified in the `model_type` parameter, and contains the parameters to be passed 
#' to the respective model. Default = NULL.
#' @param remove_obs A logical value to remove observations with categorical predictors from the test/validation set
#'                   that have not been observed during model training. Some algorithms may produce an error if this occurs. Default = FALSE.
#' @param save_models A logical value to save models during train-test splitting and/or k-fold cross validation. Default = FALSE.
#' @param save_data A logical value to save all training and test/validation sets during train-test splitting and/or k-fold cross validation. Default = FALSE.
#' @param final_model A logical value to use all complete observations in the input data for model training. Default = FALSE.
#' @param n_cores A numerical value specifying the number of cores to use for parallel processing. Default = NULL.
#' @param standardize A logical value or numerical vector. If TRUE, all columns except the target, columns of class character, and columns of class factor, will be standardized. To specify the columns to be standardized, create a numerical or character vector consisting of the column indices or names to be standardized.
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
#'   - "gbm": distribution, weights, var.monotone, n.trees, interaction.depth, n.minobsinnode, shrinkage, train.faction, cv.folds, keep.data, verbose, class.stratify.cv, n.cores
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
#' @return A list containing the results of train-test splitting and/or k-fold cross-validation (if specified), performance metrics, information on the class distribution in the training, test sets, and folds (if applicable), 
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
                    mod_args = NULL, remove_obs = FALSE, save_models = FALSE, save_data = FALSE, final_model = FALSE, n_cores = NULL, standardize = NULL, ...){
  
  # Ensure model type is lowercase
  if(!is.null(model_type)) model_type <- tolower(model_type)
  
  
  # Checking if inputs are valid
  vswift:::.error_handling(data = data, target = target, predictors = predictors, n_folds = n_folds, split = split, model_type = model_type, threshold = threshold, stratified = stratified, random_seed = random_seed, 
                           impute_method = impute_method, impute_args = impute_args, mod_args = mod_args, n_cores = n_cores, standardize = standardize, call = "classCV", ...)
  # Ensure model types are unique
  model_type <- unique(model_type)
  
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
  } else{
    data_levels <- NA
  }
  
  # Remove missing data if no imputation specified and remove rows with missing target variables.
  if(is.null(impute_method)){
    preprocessed_data <- vswift:::.remove_missing_data(data = data)
    
    rownames(preprocessed_data) <- 1:nrow(preprocessed_data)
    
    # Store information
    classCV_output <- vswift:::.store_parameters(data = data, preprocessed_data = preprocessed_data, predictor_vec = predictor_vec, target = target, model_type = model_type,
                                                 threshold = threshold, split = split, n_folds = n_folds, stratified = stratified, random_seed = random_seed, mod_args = mod_args, ...)
    override_imputation <- NULL
    
  } else {
    preprocessed_data <- vswift:::.remove_missing_target(data = data, target = target)
    
    # Check if removing missing target variables removes all missing data
    override_imputation <- vswift:::.check_if_missing(data = preprocessed_data)

    
    # Store information
    classCV_output <- vswift:::.store_parameters(data = data, preprocessed_data = preprocessed_data, predictor_vec = predictor_vec, target = target, model_type = model_type,
                                                 threshold = threshold, split = split, n_folds = n_folds, stratified = stratified, random_seed = random_seed, mod_args = mod_args, parallel = TRUE, n_cores = n_cores, ...)
  }
  
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
    # Assignments to iterator variable
    iterator_vector <- "Training"
    
  }
  # Adding information to data frame
  if(!is.null(n_folds)){
    # Create folds; start with randomly shuffling indices
    indices <- sample(1:nrow(preprocessed_data),nrow(preprocessed_data))
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
  # Expand dataframe
  
  classCV_output <- vswift:::.expand_dataframe(classCV_output = classCV_output, split = split, n_folds = n_folds, model_type = model_type)
  
  # Save data
  if(save_data == TRUE) classCV_output[["saved_data"]][["preprocessed_data"]] <- preprocessed_data
  

  # Move logistic and gbm to back of list
  if(all(length(model_type) > 1, any(c("logistic","gbm") %in% model_type))){
    x <- which(model_type %in% c("logistic","gbm"))
    model_type <- c(model_type[-x],model_type[x])
  }
  
  # Imputation
  if(all(!is.null(impute_method), override_imputation == FALSE)){
    # Add information to output
    classCV_output <- vswift:::.store_parameters(impute_method = impute_method, impute_args = impute_args, classCV_output = classCV_output)
    processed_data_list <- list()
    # Imputation; Create processed data list so each model type uses the same imputated dataset
    for(i in iterator_vector){
      imputation_output <- vswift:::.imputation(preprocessed_data = preprocessed_data, imputation_method = impute_method, impute_args = impute_args, classCV_output = classCV_output, iteration = i, parallel = FALSE)
      classCV_output <- imputation_output[["classCV_output"]]
      processed_data_list[[i]] <- imputation_output[["processed_data"]]
    }
    if(final_model == TRUE){
      imputation_output <- vswift:::.imputation(preprocessed_data = preprocessed_data, imputation_method = impute_method, impute_args = impute_args, classCV_output = classCV_output, final = TRUE)
      classCV_output <- imputation_output[["classCV_output"]]
      processed_data_list[["final model"]] <- imputation_output[["processed_data"]]
    }
  }
  
  # Variable to keep track of logistic and gbm
  model_eval_tracker <- 1
  # Perform model evaluation
  for(model_name in model_type){
    # Convert target variable
    if(all(model_name %in% c("logistic","gbm"), model_eval_tracker == 1)){
        if(all(!is.null(impute_method), override_imputation == FALSE)){
          if(final_model == TRUE){
            new_iterator_vector <- c(iterator_vector, "final model")
          } else{
            new_iterator_vector <- iterator_vector
          }
          for(i in new_iterator_vector){
            processed_data_list[[i]][,target] <- sapply(processed_data_list[[i]][,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
          }
        } else{
          preprocessed_data[,target] <- sapply(preprocessed_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]])
        }
      model_eval_tracker <- model_eval_tracker + 1
    }
    
    # Check if split or cv specified then perform evaluation
    if(any(!is.null(split), !is.null(n_folds))){
      if(is.null(n_cores)){
        for(i in iterator_vector){
          if(all(!is.null(impute_method), override_imputation == FALSE)){
            processed_data <- processed_data_list[[i]]
          } else {
            processed_data <- preprocessed_data
          }
          
          
          validation_output <- vswift:::.validation(i = i, model_name = model_name, preprocessed_data = processed_data, 
                                                    data_levels = data_levels, formula = formula, target = target, predictors = predictors, split = split, 
                                                    n_folds = n_folds, mod_args = mod_args, remove_obs = remove_obs, save_data = save_data, 
                                                    save_models = save_models, classCV_output = classCV_output, threshold = threshold, standardize = standardize, parallel = FALSE, ...)
          
          classCV_output <- validation_output
        }
      } else {
        registerDoParallel(n_cores)
        parallel_output <- foreach(i = iterator_vector, .combine = "c") %dopar% {
          
          if(all(!is.null(impute_method), override_imputation == FALSE)){
            processed_data <- processed_data_list[[i]]
          } else {
            processed_data <- preprocessed_data
            
          }
          
          output <- classCV_output
          
          vswift:::.validation(i = i, model_name = model_name, preprocessed_data = processed_data, 
                               data_levels = data_levels, formula = formula, target = target, predictors = predictors, split = split, 
                               n_folds = n_folds, mod_args = mod_args, remove_obs = remove_obs, save_data = save_data,  
                               save_models = save_models, classCV_output = classCV_output, threshold = threshold, standardize = standardize, parallel = TRUE,  ...)
          
        }
        if(!is.null(n_cores)){
          # Stop cluster
          stopImplicitCluster()
          # Separate lists
          list_length <- length(parallel_output)/length(iterator_vector)
          parallel_output  <- split(parallel_output, rep(1:length(iterator_vector), each = list_length))
          # Change names of sublist to names in iterator vector
          names(parallel_output) <- iterator_vector
          classCV_output <- vswift:::.merge_list(save_data = save_data, save_models = save_models, model_name = model_name, parallel_list = parallel_output, preprocessed_data = preprocessed_data, impute_method = impute_method)
        }
      }
      # Calculate mean, standard deviation, and standard error for cross validation
      if(!is.null(n_folds)){
        idx <- nrow(classCV_output[["metrics"]][[model_name]][["cv"]])
        classCV_output[["metrics"]][[model_name]][["cv"]][(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
        # Calculate mean, standard deviation, and sd for each column except for fold
        for(colname in colnames(classCV_output[["metrics"]][[model_name]][["cv"]] )[colnames(classCV_output[["metrics"]][[model_name]][["cv"]] ) != "Fold"]){
          # Create vector containing corresponding column name values for each fold
          num_vector <- classCV_output[["metrics"]][[model_name]][["cv"]][1:idx, colname]
          classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Mean CV:"),colname] <- mean(num_vector, na.rm = T)
          classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector, na.rm = T)
          classCV_output[["metrics"]][[model_name]][["cv"]][which(classCV_output[["metrics"]][[model_name]][["cv"]]$Fold == "Standard Error CV:"),colname] <- sd(num_vector, na.rm = T)/sqrt(n_folds)
        }
      }
    }
    
    # Generate final model
    if(final_model == TRUE){
      if(any(is.null(impute_method), override_imputation == TRUE)){
        processed_data_list <- list()
        processed_data_list[["final model"]] <- preprocessed_data
      }
      # Generate model depending on chosen model_type
      classCV_output[["final model"]][[model_name]]  <- vswift:::.generate_model(model_type = model_name, formula = formula, predictors = predictors, target = target, model_data = processed_data_list[["final model"]], mod_args = mod_args, ...)
    }
  }
  
  # Make list a vswift class
  class(classCV_output) <- "vswift"
  return(classCV_output)
}