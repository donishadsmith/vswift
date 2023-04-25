stratified_split <- function(data = NULL, y_col = NULL, split = NULL, fold_n = NULL, stratified = FALSE, random_seed = NULL,create_data = TRUE){
  #Check input
  .error_handling(data = data, y_col = y_col, fold_n = fold_n, split = split, stratified = stratified, random_seed = random_seed, call = "stratified_split")
  #Set seed
  if(!is.null(random_seed)){
    set.seed(random_seed)
  }
  #Initialize stratified list for out
  output <- list()
  #Stratified splitting
  if(stratified == TRUE){
    #Get column name
    if(is.numeric(y_col)){
      y_col <- colnames(data)[y_col]
    }
    #Isolate stratified variable
    stratify_var <- factor(data[,y_col])
    #Get classes
    output[["classes"]][[y_col]] <- names(table(data[,y_col]))
    #Get Proportions
    output[["class_proportions"]] <- table(data[,y_col])/sum(table(data[,y_col]))
    #Get indices of classes
    for(class in as.character(output[["classes"]][[y_col]])){
      output[["class_indices"]][[class]] <- which(stratify_var == class)
    }
    if(!is.null(split)){
      #Create class indices variable
      class_indices <- output[["class_indices"]]
      #Split sizes
      training_n <- nrow(data)*split
      test_n <- nrow(data) - (nrow(data)*split)
      #Initialize list
      output[["sample_indices"]][["split"]] <- list()
      output[["sample_proportions"]][["split"]] <- list()
      for(class in as.character(output[["classes"]][[y_col]])){
        #Check if sampling possible
        .stratified_check(class = class, class_indices = class_indices, output = output, n = training_n)
        #Store indices for training set
        output[["sample_indices"]][["split"]][["training"]] <- c(output[["sample_indices"]][["split"]][["training"]] ,sample(class_indices[[class]],size = round(training_n*output[["class_proportions"]][[class]],0), replace = F))
        #Remove indices to not add to test set
        class_indices[[class]] <- class_indices[[class]][!(class_indices[[class]] %in% output[["sample_indices"]][["split"]][["training"]])]
        #Check if sampling possible
        .stratified_check(class = class, class_indices = class_indices, output = output, n = test_n)
        #Add indices for test set
        output[["sample_indices"]][["split"]][["test"]] <- c(output[["sample_indices"]][["split"]][["test"]] ,sample(class_indices[[class]],size = round(test_n*output[["class_proportions"]][[class]],0), replace = F))
      }
      #Get proportions
      output[["sample_proportions"]][["split"]][["training"]] <- table(stratify_var[output[["sample_indices"]][["split"]][["training"]]])/sum(table(stratify_var[output[["sample_indices"]][["split"]][["training"]]]))
      output[["sample_proportions"]][["split"]][["test"]] <- table(stratify_var[output[["sample_indices"]][["split"]][["test"]]])/sum(table(stratify_var[output[["sample_indices"]][["split"]][["test"]]]))
      if(create_data == TRUE){
        output[["data"]][["split"]] <- list()
        #Split data
        output[["data"]][["split"]][["training"]] <- data[output[["sample_indices"]][["split"]][["training"]],]
        output[["data"]][["split"]][["test"]] <- data[output[["sample_indices"]][["split"]][["test"]],]
      }
    }
    if(!is.null(fold_n)){
      #Create class indices variable
      class_indices <- output[["class_indices"]]
      #Initialize list to store indices, proportions, and data
      output[["sample_indices"]][["cv"]] <- list()
      output[["sample_proportions"]][["cv"]] <- list()
      if(create_data == TRUE){
        output[["data"]][["cv"]] <- list()
      }
      for(i in 1:fold_n){
        #Keep initializing variable
        fold_idx <- c()
        #fold size; try to undershoot for excess
        fold_size <- floor(nrow(data)/fold_n)
        for(class in as.character(output[["classes"]][[y_col]])){
          #Check if sampling possible
          .stratified_check(class = class, class_indices = class_indices, output = output, n = fold_size)
          #Check if sampling possible
          fold_idx <- c(fold_idx, sample(class_indices[[class]],size = floor(fold_size*output[["class_proportions"]][[class]]), replace = F))
          #Remove already selected indices
          class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
        }
        #Add indices to list
        output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        #Update proportions
        output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
      }
      #Deal with excess indices
      excess <- nrow(data) - length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
      if(excess > 0){
        for(class in names(output[["class_proportions"]])){
          fold_idx <- class_indices[[class]]
          if(length(fold_idx) > 0){
            leftover <- rep(1:fold_n,length(fold_idx))[1:length(fold_idx)]
            for(i in 1:length(leftover)){
              #Add indices to list
              output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
              #Update class proportions
              output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(stratify_var[output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
            }
          }
        }
      }
      if(create_data == TRUE){
        #Split data
        for(i in 1:fold_n){
          output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]],]
        }
      }
    }
  }else{
    if(!is.null(split)){
      #Initialize list
      output[["sample_indices"]][["split"]] <- list()
      #Create test and training set
      output[["sample_indices"]][["split"]][["training"]] <- sample(1:nrow(data),size = round(nrow(data)*split,0),replace = F)
      output[["sample_indices"]][["split"]][["test"]] <- c(1:nrow(data))[-output[["sample_indices"]][["split"]][["training"]]]
      if(create_data == TRUE){
        output[["data"]][["split"]] <- list()
        output[["data"]][["split"]][["training"]] <- data[output[["sample_indices"]][["split"]][["training"]],]
        output[["data"]][["split"]][["test"]] <- data[output[["sample_indices"]][["split"]][["test"]],]
      }
    }
    if(!is.null(fold_n)){
      #Create folds; start with randomly shuffling indices
      indices <- sample(1:nrow(data))
      #Get floor
      fold_size_vector <- rep(floor(nrow(data)/fold_n),fold_n)
      excess <- nrow(data) - sum(fold_size_vector)
      if(excess > 0){
        folds_vector <- rep(1:fold_n,excess)[1:excess]
        for(num in folds_vector){
          fold_size_vector[num] <- fold_size_vector[num] + 1
        }
      }
      #random shuffle
      fold_size_vector <- sample(fold_size_vector, size = length(fold_size_vector), replace = FALSE)
      for(i in 1:fold_n){
        #Create fold with stratified or non stratified sampling
        fold_idx <- indices[1:fold_size_vector[i]]
        #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
        indices <- indices[-c(1:fold_size_vector[i])]
        #Add indices to list
        output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
        if(create_data == TRUE){
          output[["data"]][["cv"]][[sprintf("fold %s",i)]] <- data[fold_idx,]
        }
      }
    }
  }
  return(output)
}




