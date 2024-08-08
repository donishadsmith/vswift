#Helper function for classCV for stratified sampling
#' @noRd
#' @export
.stratified_sampling <- function(data, type, output, target, split = NULL, k = NULL,random_seed = NULL){
  switch(type,
         "split" = {
           # Set seed
           if(!is.null(random_seed)) set.seed(random_seed)
           # Get class indices
           class_indices <- output[["class_indices"]]
           # Split sizes
           training_n <- round(nrow(data)*split,0)
           test_n <- nrow(data) - training_n
           # Initialize list
           output[["sample_indices"]][["split"]] <- list()
           output[["sample_proportions"]][["split"]] <- list()
           # Extract indices for each class
           for(class in names(output[["class_proportions"]])){
             # Check if sampling possible
             .stratified_check(class = class, class_indices = class_indices, output = output, n = training_n)
             # Store indices for training set
             output[["sample_indices"]][["split"]][["training"]] <- c(
               output[["sample_indices"]][["split"]][["training"]],
               sample(
                 class_indices[[class]],size = round(training_n*output[["class_proportions"]][[class]], 0), replace = F
                 )
               )
             # Remove indices to not add to test set
             class_indices[[class]] <- class_indices[[class]][
               !(class_indices[[class]] %in% output[["sample_indices"]][["split"]][["training"]])
               ]
             # Check if sampling possible
             .stratified_check(class = class, class_indices = class_indices, output = output, n = test_n)
             # Add indices for test set
             output[["sample_indices"]][["split"]][["test"]] <- c(
               output[["sample_indices"]][["split"]][["test"]],
               sample(class_indices[[class]], size = round(test_n*output[["class_proportions"]][[class]], 0),
                      replace = F
                      )
               )
           }
           # Store proportions of data in training set
           training_subgroups_n <- table(data[,target][output[["sample_indices"]][["split"]][["training"]]])
           training_n <- sum(table(data[,target][output[["sample_indices"]][["split"]][["training"]]]))
           output[["sample_proportions"]][["split"]][["training"]] <- training_subgroups_n/training_n
           # Store proportions of data  in test set
           test_subgroups_n <- table(data[,target][output[["sample_indices"]][["split"]][["test"]]])
           test_n <- sum(table(data[,target][output[["sample_indices"]][["split"]][["test"]]]))
           output[["sample_proportions"]][["split"]][["test"]] <- test_subgroups_n/test_n
           # Output
           stratified_sampling_output <- list("output" = output)
         },
         "k-fold" = {
           # Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }

           # Get class indices
           class_indices <- output[["class_indices"]]
           #Initialize sample_indices for cv since it will be three levels
           output[["sample_indices"]][["cv"]] <- list()

           # Create folds
           for(i in 1:k){
             # Keep initializing variable
             fold_idx <- c()
             # Fold size; try to undershoot for excess
             fold_size <- floor(nrow(data)/k)
             # Assign class indices to each fold
             for(class in names(output[["class_proportions"]])){
               # Check if sampling possible
               .stratified_check(class = class, class_indices = class_indices, output = output, n = fold_size)
               # Check if sampling possible
               fold_idx <- c(fold_idx,
                             sample(class_indices[[class]],
                                    size = floor(fold_size*output[["class_proportions"]][[class]]), replace = F))
               # Remove already selected indices
               class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
             }
             # Add indices to list
             output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
             # Update proportions
             subgroups_n <- table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])
             n <- sum(table(data[,target][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
             output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- subgroups_n/n
           }
           # Deal with excess indices
           excess <- nrow(data) - length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
           if(excess > 0){
             for(class in names(output[["class_proportions"]])){
               fold_idx <- class_indices[[class]]
               if(length(fold_idx) > 0){
                 leftover <- rep(1:k,length(fold_idx))[1:length(fold_idx)]
                 for(i in 1:length(leftover)){
                   # Add indices to list
                   fold <- sprintf("fold %s",leftover[i])
                   output[["sample_indices"]][["cv"]][[fold]] <- c(fold_idx[i],
                                                                   output[["sample_indices"]][["cv"]][[fold]])
                   # Update class proportions
                   subgroup_n <- table(data[,target][output[["sample_indices"]][["cv"]][[fold]]])
                   n <- sum(table(data[,target][output[["sample_indices"]][["cv"]][[fold]]]))
                   output[["sample_proportions"]][["cv"]][[fold]] <- subgroup_n/n
                 }
               }
             }
           }
           # Output
           stratified_sampling_output <- list("output" = output)
         }
  )
  # Return output
  return(stratified_sampling_output)
}

# Helper function for .stratified_sampling to error check
#' @noRd
#' @export
.stratified_check <- function(class, class_indices, output, n){
  # Check if there are zero indices for a specific class
  if(round(n*output[["class_proportions"]][[class]], 0) == 0){
    stop(sprintf("0 indices selected for %s class\n not enough samples for stratified sampling", class))
  }
  # Check if there there are enough indices in class for proper assignment
  if(round(n*output[["class_proportions"]][[class]], 0) > length(class_indices[[class]])){
    stop(sprintf("not enough samples of %s class for stratified sampling", class))
  }
}
