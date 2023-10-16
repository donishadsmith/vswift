#' Create split datasets and/or folds with optional stratification
#' 
#' `genFolds` generates train-test split datasets and/or k-fold cross-validation folds, with the option to perform stratified sampling based on class distribution.
#' 
#' 
#' @param data A data frame.
#' @param target A numerical index or character name for the target variable. Only needs to be specified if stratified = TRUE. Default = NULL.
#' @param split A numerical value between 0.5 to 0.9 indicating the proportion of data to use for the training set, leaving the rest for the test set. If not specified, train-test splitting will not be done.
#' @param n_folds A numerical value between 3-30 indicating the number of k-folds. If left empty, k-fold cross validation will not be performed.
#' @param stratified A logical value indicating if stratified sampling should be used. Default = FALSE.
#' @param random_seed A numerical value for the random seed to be used. Default = NULL.
#' @param create_data A logical value indicating whether to create all training and test/validation data frames. Default = FALSE. 
#' @return A list containing the indices for train-test splitting and/or k-fold cross-validation, with information on the class distribution in the training, test sets, and folds (if applicable)
#'         as well as the generated split datasets and folds based on the indices.
#' @examples
#' # Load example dataset 
#' 
#' data(iris)
#' 
#' # Obtain indices for 80% training/test split and 5-fold CV
#' 
#' output <- genFolds(data = iris, target = "Species", split = 0.8, n_folds = 5)
#'
#' @author Donisha Smith
#' @export

genFolds <- function(data, target = NULL,  split = NULL, n_folds = NULL, stratified = FALSE, random_seed = NULL, create_data = FALSE){
  # Check input
  vswift:::.error_handling(data = data, target = target, n_folds = n_folds, split = split, stratified = stratified, random_seed = random_seed, call = "stratified_split")
  # Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  # Initialize stratified list for out
  output <- list()
  # Stratified splitting
  if(stratified == TRUE){
    # Get column name
    if(is.numeric(target)){
      target <- colnames(data)[target]
    }
    # Isolate stratified variable
    stratify_var <- factor(data[,target])
    # Get classes
    output[["classes"]][[target]] <- names(table(data[,target]))
    # Get proportions
    output[["class_proportions"]] <- table(data[,target])/sum(table(data[,target]))
    # Get indices of classes
    for(class in as.character(output[["classes"]][[target]])){
      output[["class_indices"]][[class]] <- which(stratify_var == class)
    }
    if(!is.null(split)){
      # Create separate class indices variable to delete selected indices
      class_indices <- output[["class_indices"]]
      # Split sizes
      training_n <- round(nrow(data)*split,0)
      test_n <- nrow(data) - training_n
      # Initialize list
      output[["sample_indices"]][["split"]] <- list()
      output[["sample_proportions"]][["split"]] <- list()
      for(class in as.character(output[["classes"]][[target]])){
        # Check if sampling possible
        vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = training_n)
        # Store indices for training set
        output[["sample_indices"]][["split"]][["training"]] <- c(output[["sample_indices"]][["split"]][["training"]] ,sample(class_indices[[class]],size = round(training_n*output[["class_proportions"]][[class]],0), replace = F))
        # Remove indices to not add to test set
        class_indices[[class]] <- class_indices[[class]][!(class_indices[[class]] %in% output[["sample_indices"]][["split"]][["training"]])]
        # Check if sampling possible
        vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = test_n)
        # Add indices for test set
        output[["sample_indices"]][["split"]][["test"]] <- c(output[["sample_indices"]][["split"]][["test"]] ,sample(class_indices[[class]],size = round(test_n*output[["class_proportions"]][[class]],0), replace = F))
      }
      # Get proportions
      output[["sample_proportions"]][["split"]][["training"]] <- table(stratify_var[output[["sample_indices"]][["split"]][["training"]]])/sum(table(stratify_var[output[["sample_indices"]][["split"]][["training"]]]))
      output[["sample_proportions"]][["split"]][["test"]] <- table(stratify_var[output[["sample_indices"]][["split"]][["test"]]])/sum(table(stratify_var[output[["sample_indices"]][["split"]][["test"]]]))
      if(create_data == TRUE){
        output[["data"]][["split"]] <- list()
        # Split data
        output[["data"]][["split"]][["training"]] <- data[output[["sample_indices"]][["split"]][["training"]],]
        output[["data"]][["split"]][["test"]] <- data[output[["sample_indices"]][["split"]][["test"]],]
      }
    }
    if(!is.null(n_folds)){
      # Create class indices variable
      class_indices <- output[["class_indices"]]
      # Initialize list to store indices, proportions, and data
      output[["sample_indices"]][["cv"]] <- list()
      output[["sample_proportions"]][["cv"]] <- list()
      if(create_data == TRUE){
        output[["data"]][["cv"]] <- list()
      }
      for(i in 1:n_folds){
        # Keep initializing variable
        fold_idx <- c()
        # fold size; try to undershoot for excess
        fold_size <- floor(nrow(data)/n_folds)
        for(class in as.character(output[["classes"]][[target]])){
          # Check if sampling possible
          vswift:::.stratified_check(class = class, class_indices = class_indices, output = output, n = fold_size)
          # Check if sampling possible
          fold_idx <- c(fold_idx, sample(class_indices[[class]],size = floor(fold_size*output[["class_proportions"]][[class]]), replace = F))
          # Remove already selected indices
          class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
        }
        # Add indices to list
        output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        # Update proportions
        output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
      }
      # Deal with excess indices
      excess <- nrow(data) - length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
      if(excess > 0){
        for(class in names(output[["class_proportions"]])){
          fold_idx <- class_indices[[class]]
          if(length(fold_idx) > 0){
            leftover <- rep(1:n_folds,length(fold_idx))[1:length(fold_idx)]
            for(i in 1:length(leftover)){
              # Add indices to list
              output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
              # Update class proportions
              output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
            }
          }
        }
      }
      if(create_data == TRUE){
        # Split data
        for(i in 1:n_folds){
          output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]],]
        }
      }
    }
  } else {
    if(!is.null(split)){
      # Initialize list
      output[["sample_indices"]][["split"]] <- list()
      # Create test and training set
      output[["sample_indices"]][["split"]][["training"]] <- sample(1:nrow(data),size = round(nrow(data)*split,0),replace = F)
      output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(data))[-output[["sample_indices"]][["split"]][["training"]]]
      if(create_data == TRUE){
        output[["data"]][["split"]] <- list()
        output[["data"]][["split"]][["training"]] <- data[output[["sample_indices"]][["split"]][["training"]],]
        output[["data"]][["split"]][["test"]] <- data[output[["sample_indices"]][["split"]][["test"]],]
      }
    }
    if(!is.null(n_folds)){
      # Create folds; start with randomly shuffling indices
      indices <- sample(1:nrow(data))
      # Get floor
      fold_size_vector <- rep(floor(nrow(data)/n_folds),n_folds)
      excess <- nrow(data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:n_folds,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      # random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:n_folds){
        # Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        # Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        # Add indices to list
        output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        if(create_data == TRUE){
          output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[fold_idx,]
        }
      }
    }
  }
  return(output)
}




