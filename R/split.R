split <- function(data = NULL, y_col = NULL, split = NULL, k = NULL, stratified = FALSE, random_seed = NULL){
  
  .error.handling(data = data, y_col = y_col, k = k, split = split, stratified = stratified, random_seed = random_seed, call = "split")
  #Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  #Initialize stratifiedlist for out
  split_output <- list()
  #Stratified splitting
  if(stratified == TRUE){
    #Get column name
    if(is.numeric(y_col)){
      y_col <- colnames(data)[y_col]
    }
    #Isolate stratified variable
    stratify_var <- factor(data[,y_col])
    #Get classes
    split_output[["classes"]][[y_col]] <- names(table(data[,y_col]))
    #Get Proportions
    split_output[["class_proportions"]] <- table(data[,y_col])/sum(table(data[,y_col]))
    #Get indices of classes
    for(class in as.character(split_output[["classes"]][[y_col]])){
      split_output[["class_indices"]][[class]] <- which(stratify_var == class)
    }
    if(!is.null(split)){
      #Create class indices variable
      class_indices <- split_output[["class_indices"]]
      #Split sizes
      training_n <- nrow(data)*split
      test_n <- nrow(data) - (nrow(data)*split)
      for(class in as.character(split_output[["classes"]][[y_col]])){
        #Store indices for training set
        split_output[["sample_indices"]][["training"]] <- c(split_output[["sample_indices"]][["training"]] ,sample(class_indices[[class]],size = round(training_n*split_output[["class_proportions"]][[class]],0), replace = F))
        #Remove indices to not add to test set
        class_indices[[class]] <- class_indices[[class]][!(class_indices[[class]] %in% split_output[["sample_indices"]][["training"]])]
        #Add indices for test set
        split_output[["sample_indices"]][["test"]] <- c(split_output[["sample_indices"]][["test"]] ,sample(class_indices[[class]],size = round(test_n*split_output[["class_proportions"]][[class]],0), replace = F))
      }
      #Get proportions
      split_output[["sample_proportions"]][["training"]] <- table(stratify_var[split_output[["sample_indices"]][["training"]]])/sum(table(stratify_var[split_output[["sample_indices"]][["training"]]]))
      split_output[["sample_proportions"]][["test"]] <- table(stratify_var[split_output[["sample_indices"]][["test"]]])/sum(table(stratify_var[split_output[["sample_indices"]][["test"]]]))
      #Split data
      split_output[["data"]][["training"]] <- data[split_output[["sample_indices"]][["training"]],]
      split_output[["data"]][["test"]] <- data[split_output[["sample_indices"]][["test"]],]
    }
    if(!is.null(k)){
      #Create class indices variable
      class_indices <- split_output[["class_indices"]]
      #Initialize list to store indices, proportions, and data
      split_output[["sample_indices"]][["cv"]] <- list()
      split_output[["sample_proportions"]][["cv"]] <- list()
      split_output[["data"]][["cv"]] <- list()
      for(i in 1:k){
        #Keep initializing variable
        fold_idx <- c()
        #fold size; try to undershoot for excess
        fold_size <- floor(nrow(data)/k)
        for(class in as.character(split_output[["classes"]][[y_col]])){
          fold_idx <- c(fold_idx, sample(class_indices[[class]],size = floor(fold_size*split_output[["class_proportions"]][[class]]), replace = F))
          #Remove already selected indices
          class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
        }
        #Add indices to list
        split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        #Update proportions
        split_output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(stratify_var[split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(stratify_var[split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
      }
      #Deal with excess indices
      excess <- nrow(data) - length(as.numeric(unlist(split_output[["sample_indices"]][["cv"]])))
      if(excess > 0){
        for(class in names(split_output[["class_proportions"]])){
          fold_idx <- class_indices[[class]]
          if(length(fold_idx) > 0){
            leftover <- rep(1:k,length(fold_idx))[1:length(fold_idx)]
            for(i in 1:length(leftover)){
              #Add indices to list
              split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
              #Update class proportions
              split_output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(stratify_var[split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(stratify_var[split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
            }
          }
        }
      }
      #Split data
      for(i in 1:k){
        split_output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]],]
      }
    }
  }else{
    if(!is.null(split)){
      #Create test and training set
      split_output[["sample_indices"]][["training"]] <- sample(1:nrow(data),size = round(nrow(data)*split,0),replace = F)
      split_output[["data"]][["training"]] <- data[split_output[["sample_indices"]][["training"]],]
      split_output[["sample_indices"]][["test"]] <- c(1:nrow(data))[-split_output[["sample_indices"]][["training"]]]
      split_output[["data"]][["test"]] <- data[split_output[["sample_indices"]][["test"]],]
    }
    if(!is.null(k)){
      #Create folds; start with randomly shuffling indices
      indices <- sample(1:nrow(data))
      #Get floor
      fold_size_vector <- rep(floor(nrow(data)/k),k)
      excess <- nrow(data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:k,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      #random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:k){
        #Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        #Add indices to list
        split_output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        split_output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[fold_idx,]
      }
    }
  }
  return(split_output)
}





