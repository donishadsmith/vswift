#' Perform Train-Test Split and/or K-Fold Cross-Validation with optional stratified sampling for classification data
#'
#' @name classCV 
#' @description performs a train-test split and/or k-fold cross validation
#' on classification data using various classification algorithms.
#'
#' @param data A data frame containing the dataset. Default = \code{NULL}
#' @param formula A formula specifying the model to use. Default = \code{NULL}
#' @param target The target variable's numerical index or name in the data frame. Default = \code{NULL}.
#' @param predictors A vector of numerical indices or names for the predictors in the data frame. Default = \code{NULL}.
#'                   If not specified, all variables except the response variable will be used as predictors.
#'                   Default = \code{NULL}.
#' @param model_type A character string or list indicating the classification algorithm to use. Available options:
#'                   \code{"lda"} (Linear Discriminant Analysis), \code{"qda"} (Quadratic Discriminant Analysis), 
#'                   \code{"logistic"} (Logistic Regression), \code{"svm"} (Support Vector Machines),
#'                   \code{"naivebayes"} (Naive Bayes), \code{"ann"} (Artificial Neural Network), \code{"knn"}
#'                   (K-Nearest Neighbors), \code{"decisiontree"} (Decision Tree), \code{"randomforest"} (Random Forest),
#'                   \code{"multinom"} (Multinomial Logistic Regression), \code{"gbm"} (Gradient Boosting Machine).
#'                   \itemize{
#'                    \item For \code{"knn"}, the optimal k will be used unless specified with \code{ks}.
#'                    \item For \code{"ann"}, \code{size} must be specified as an additional argument.
#'                   }
#' @param threshold  A number from 0.3 to 0.7 indicating representing the decision boundary for logistic regression.
#'                   Default = \code{0.5}
#' @param mod_args  list of named sub-lists. Each sub-list corresponds to a model specified in the \code{model_type}
#' @param final_model A logical value to use all complete observations in the input data for model training.
#'                    Default = \code{FALSE}.
#' @param split A number from 0.5 and 0.9 for the proportion of data to use for the training set,
#'              leaving the rest for the test set. If not specified, train-test splitting will not be done.
#'              Default = \code{NULL}.
#' @param n_folds An integer from 3 and 30 for the number of folds to use. If not specified,
#'                k-fold cross validation will not be performed. Default = \code{NULL}.
#' @param stratified A logical value indicating if stratified sampling should be used. Default = \code{FALSE}.
#' @param random_seed A numerical value for the random seed to ensure random splitting is reproducible.
#'                    Default = \code{NULL}.
#' @param impute_method A character indicating the imputation method to use. Options include \code{"bag_impute"}
#'                      (Bagged Trees Imputation) and \code{"knn_impute"} (KNN Imputation).
#' @param impute_args A list specifying an additional argument for the imputation method. Below are the additional
#'                    arguments available for each imputation option.
#'                    \itemize{
#'                      \item \code{"bag_impute"}: \code{trees}
#'                      \item \code{"knn_impute"}: \code{neighbors}
#'                    }
#'                    For specific information about each parameter, please refer to the recipes documentation.
#'                    Default = \code{NULL}.
#'                  parameter, and contains the parameters to be passed to the respective model. Default = \code{NULL}.
#' @param save_models A logical value to save models during train-test splitting and/or k-fold cross validation.
#'                    Default = \code{FALSE}.
#' @param save_data A logical value to save all training and test/validation sets during train-test splitting
#'                  and/or k-fold cross validation. Default = \code{FALSE}.
#' @param n_cores A numerical value specifying the number of cores to use for parallel processing.
#'                Default = \code{NULL}.
#' @param remove_obs A logical value to remove observations with categorical predictors from the test/validation set
#'                   that have not been observed during model training. Some algorithms may produce an error if this
#'                   occurs. Default = \code{FALSE}.
#' @param standardize A logical value or numerical vector. If \code{TRUE}, all columns except the target, that are
#'                    numeric, will be standardized. To specify the columns to be standardized, create a numerical
#'                    or character vector consisting of the column indices or names to be standardized.
#' @param ... Additional arguments specific to the chosen classification algorithm.
#'            Please refer to the corresponding algorithm's documentation for additional arguments and their
#'            descriptions.
#' 
#' @section Model-specific additional arguments:
#'   Each option of \code{model_type} accepts additional arguments specific to the classification algorithm. The
#'   available arguments for each \code{model_type} are:
#'   \itemize{
#'    \item \code{"lda"}: \code{prior}, \code{method}, \code{nu}
#'    \item \code{"qda"}: \code{prior}, \code{method}, \code{nu}
#'    \item \code{"logistic"}: \code{weights}, \code{singular.ok}, \code{maxit}
#'    \item \code{"svm"}: \code{kernel}, \code{degree}, \code{gamma}, \code{cost}, \code{nu}
#'    \item \code{"naivebayes"}: \code{prior}, \code{laplace}, \code{usekernel}
#'    \item \code{"ann"}: \code{size}, \code{rang}, \code{decay}, \code{maxit}, \code{softmax},
#'                        \code{entropy}, \code{abstol}, \code{reltol}
#'    \item \code{"knn"}: \code{kmax}, \code{ks}, \code{distance}, \code{kernel}
#'    \item \code{"decisiontree"}: \code{weights}, \code{method},\code{parms}, \code{control}, \code{cost}
#'    \item \code{"randomforest"}: \code{weights}, \code{ntree}, \code{mtry}, \code{nodesize}, \code{importance}
#'    \item \code{"multinom"}: \code{weights}, \code{Hess}
#'    \item \code{"gbm"}: \code{params}, \code{nrounds}
#'   }
#'
#' @section Functions used from packages for each option for \code{model_type}:
#'   \itemize{
#'    \item \code{"lda"}: \code{lda()} from MASS package
#'    \item \code{"qda"}: \code{qda()} from MASS package
#'    \item \code{"logistic"}: \code{glm()} from base package with \code{family = "binomial"}
#'    \item \code{"svm"}: \code{svm()} from e1071 package
#'    \item \code{"naivebayes"}: \code{naive_bayes()} from naivebayes package
#'    \item \code{"ann"}: \code{nnet()} from nnet package
#'    \item \code{"knn"}: \code{train.kknn()} from kknn package
#'    \item \code{"decisiontree"}: \code{rpart()} from rpart package
#'    \item \code{"randomforest"}: \code{randomForest()} from randomForest package 
#'    \item \code{"multinom"}: \code{multinom()} from nnet package
#'    \item \code{"gbm"}: \code{xgb.train()} from xgboost package
#'   }
#'               
#' @return A list containing the results of train-test splitting and/or k-fold cross-validation (if specified),
#'         performance metrics, information on the class distribution in the training, test sets, and folds
#'         (if applicable), saved models (if specified), and saved datasets (if specified), and a final model
#'         (if specified).
#' 
#' @seealso \code{\link{print.vswift}}, \code{\link{plot.vswift}}
#' 
#' @examples
#' # Load an example dataset
#' data(iris)
#'
#' # Perform a train-test split with an 80% training set using LDA
#' result <- classCV(data = iris, target = "Species", 
#'                   split = 0.8, model_type = "lda")
#' 
#' # Print parameters and metrics
#' result
#'
#' # Perform 5-fold cross-validation using Gradient Boosted Model
#' result <- classCV(data = iris, formula = Species~., n_folds = 5, 
#'                   model_type = "gbm",
#'                   params = list(objective = "multi:softprob",
#'                                 num_class = 3,eta = 0.3,max_depth = 6), 
#'                                 nrounds = 10)
#' 
#' # Print parameters and metrics
#' result
#' 
#' 
#' # Perform 5-fold cross-validation a train-test split w/multiple models
#' 
#' args <- list("knn" = list(ks = 5), "ann" = list(size = 20))
#' result <- classCV(data = iris, target = 5, split = 0.8, 
#'                   model_type = c("decisiontree","knn", "ann","svm"), 
#'                   n_folds = 3,mod_args = args, stratified = TRUE)
#' 
#' # Print parameters and metrics
#' result
#' 
#' @author Donisha Smith
#' 
#' @importFrom future plan multisession sequential
#' @importFrom future.apply future_lapply
#' @importFrom stats as.formula complete.cases glm predict sd
#' @export
classCV <- function(data, formula = NULL, target = NULL, predictors = NULL, model_type, threshold = 0.5, mod_args = NULL, 
                    final_model = FALSE, split = NULL, n_folds = NULL, stratified = FALSE, random_seed = NULL,
                    impute_method = NULL, impute_args = NULL, save_models = FALSE, save_data = FALSE, n_cores = NULL,
                    remove_obs = FALSE, standardize = NULL, ...){
  
  # Ensure model type is lowercase
  if(!is.null(model_type)) model_type <- tolower(model_type)
  
  
  # Checking if inputs are valid
  .error_handling(formula = formula, data = data, target = target, predictors = predictors, n_folds = n_folds,
                  split = split, model_type = model_type, threshold = threshold, stratified = stratified,
                  random_seed = random_seed, impute_method = impute_method, impute_args = impute_args,
                  mod_args = mod_args, n_cores = n_cores, standardize = standardize, call = "classCV", ...)
  # Ensure model types are unique
  model_type <- unique(model_type)
  
  # Set seed
  if(!is.null(random_seed)) set.seed(random_seed)
  
  # Get feature variable and predictor variables
  get_features_target_output <- .get_features_target(formula = formula, target = target, predictors = predictors,
                                                     data = data)
  
  target <- get_features_target_output[["target"]]
  predictor_vec <- get_features_target_output[["predictor_vec"]]

  # Ensure data is factor
  create_factor_output <- .create_factor(data = data, target = target, model_type = model_type)
  data <- create_factor_output[["data"]] 
  if("svm" %in% model_type){
    data_levels <- create_factor_output[["data_levels"]]
  } else{
    data_levels <- NA
  }
  
  # Remove missing data if no imputation specified and remove rows with missing target variables.
  if(is.null(impute_method)){
    preprocessed_data <- .remove_missing_data(data = data)
    
    rownames(preprocessed_data) <- 1:nrow(preprocessed_data)
    
    # Store information
    classCV_output <- .store_parameters(formula = formula, data = data, preprocessed_data = preprocessed_data,
                                        predictor_vec = predictor_vec, target = target, model_type = model_type,
                                        threshold = threshold, split = split, n_folds = n_folds,
                                        stratified = stratified, random_seed = random_seed, mod_args = mod_args,
                                        n_cores=n_cores, ...)
    override_imputation <- NULL
    
  } else {
    preprocessed_data <- .remove_missing_target(data = data, target = target)
    
    # Check if removing missing target variables removes all missing data
    override_imputation <- .check_if_missing(data = preprocessed_data)

    
    # Store information
    classCV_output <- .store_parameters(formula = formula, data = data, preprocessed_data = preprocessed_data,
                                        predictor_vec = predictor_vec, target = target, model_type = model_type,
                                        threshold = threshold, split = split, n_folds = n_folds,
                                        stratified = stratified, random_seed = random_seed, mod_args = mod_args,
                                        n_cores = n_cores, ...)
  }
  
  # Get formula
  formula <- classCV_output[["formula"]]
  
  # Create class dictionary
  if(any(model_type %in% c("logistic", "gbm"))){
    classCV_output <- .create_dictionary(preprocessed_data = preprocessed_data, target = target,
                                         classCV_output = classCV_output)
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
      stratified.sampling_output <- .stratified_sampling(data = preprocessed_data,type = "split", split = split,
                                                         output = classCV_output, target = target,
                                                         random_seed = random_seed)
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
      stratified_sampling_output <- .stratified_sampling(data = preprocessed_data, type = "k-fold",
                                                         output = classCV_output, k = n_folds,
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
  
  classCV_output <- .expand_dataframe(classCV_output = classCV_output, split = split, n_folds = n_folds,
                                      model_type = model_type)
  
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
    classCV_output <- .store_parameters(impute_method = impute_method, impute_args = impute_args,
                                        classCV_output = classCV_output)
    processed_data_list <- list()
    # Imputation; Create processed data list so each model type uses the same imputated dataset
    for(i in iterator_vector){
      imputation_output <- .imputation(preprocessed_data = preprocessed_data, target = target, predictors = predictors,
                                       formula = formula, imputation_method = impute_method, impute_args = impute_args,
                                       classCV_output = classCV_output, iteration = i, parallel = FALSE,
                                       random_seed = random_seed)
      classCV_output <- imputation_output[["classCV_output"]]
      processed_data_list[[i]] <- imputation_output[["processed_data"]]
    }
    if(final_model == TRUE){
      imputation_output <- .imputation(preprocessed_data = preprocessed_data, target = target, predictors = predictors,
                                       formula = formula, imputation_method = impute_method, impute_args = impute_args,
                                       classCV_output = classCV_output, final = TRUE, random_seed = random_seed)
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
            processed_data_list[[i]][,target] <- sapply(
              processed_data_list[[i]][,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]]
              )
          }
        } else{
          preprocessed_data[,target] <- sapply(
            preprocessed_data[,target], function(x) classCV_output[["class_dictionary"]][[as.character(x)]]
            )
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
          
          
          validation_output <- .validation(i = i, model_name = model_name, preprocessed_data = processed_data,
                                           stratified = stratified, data_levels = data_levels, formula = formula,
                                           target = target, predictors = predictors, split = split, n_folds = n_folds,
                                           mod_args = mod_args, remove_obs = remove_obs, save_data = save_data, 
                                           save_models = save_models, classCV_output = classCV_output,
                                           threshold = threshold, standardize = standardize, parallel = FALSE,
                                           random_seed = random_seed, ...)
          
          classCV_output <- validation_output
        }
      } else {
        plan(multisession, workers = n_cores)
        parallel_output <- future_lapply(iterator_vector, function(i){
          
          if(all(!is.null(impute_method), override_imputation == FALSE)){
            processed_data <- processed_data_list[[i]]
          } else {
            processed_data <- preprocessed_data
            
          }
          
          output <- classCV_output
          
          .validation(i = i, model_name = model_name, preprocessed_data = processed_data, stratified = stratified,
                      data_levels = data_levels, formula = formula, target = target, predictors = predictors,
                      split = split, n_folds = n_folds, mod_args = mod_args, remove_obs = remove_obs,
                      save_data = save_data, save_models = save_models, classCV_output = classCV_output,
                      threshold = threshold, standardize = standardize, parallel = TRUE, random_seed = random_seed, ...)
          
        },future.seed=if(!is.null(random_seed)) random_seed else TRUE)
        
        # Close the background workers
        plan(sequential)
        # Merge results
        names(parallel_output) <- iterator_vector
        classCV_output <- .merge_list(save_data = save_data, save_models = save_models, model_name = model_name,
                                      parallel_list = parallel_output, processed_data = processed_data,
                                      impute_method = impute_method)
      }
      # Calculate mean, standard deviation, and standard error for cross validation
      if(!is.null(n_folds)){
        temp_df <- classCV_output[["metrics"]][[model_name]][["cv"]]
        idx <- nrow(temp_df)
        new_row <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
        temp_df[(idx + 1):(idx + 3),"Fold"] <- new_row
        # Calculate mean, standard deviation, and sd for each column except for fold
        for(colname in colnames(temp_df)[colnames(temp_df) != "Fold"]){
          # Create vector containing corresponding column name values for each fold
          num_vector <- temp_df[1:idx, colname]
          temp_df[which(temp_df$Fold == "Mean CV:"),colname] <- mean(num_vector, na.rm = TRUE)
          temp_df[which(temp_df$Fold == "Standard Deviation CV:"),colname] <- sd(num_vector, na.rm = TRUE)
          temp_df[which(temp_df$Fold == "Standard Error CV:"),colname] <- sd(num_vector, na.rm = TRUE)/sqrt(n_folds)
        }
        # Reassign
        classCV_output[["metrics"]][[model_name]][["cv"]] <- temp_df
      }
    }
    
    # Generate final model
    if(final_model == TRUE){
      if(any(is.null(impute_method), override_imputation == TRUE)){
        processed_data_list <- list()
        processed_data_list[["final model"]] <- preprocessed_data
      }
      # Generate model depending on chosen model_type
      classCV_output[["final model"]][[model_name]]  <- .generate_model(model_type = model_name, formula = formula,
                                                                        predictors = predictors, target = target,
                                                                        model_data = processed_data_list[["final model"]],
                                                                        mod_args = mod_args,
                                                                        random_seed = random_seed,
                                                                        ...)
    }
  }
  
  # Make list a vswift class
  class(classCV_output) <- "vswift"
  return(classCV_output)
}
