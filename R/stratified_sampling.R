.stratified.sampling <- function(data,type, output, response_var, split = NULL, fold_size = NULL, k = NULL,
                                 class_indices = NULL, random_seed = NULL){
  switch(type,
         "split" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           #Get proportions
           output[["class_proportions"]] <- table(data[,response_var])/sum(table(data[,response_var]))
           #Split sizes
           training_n <- nrow(data)*split
           test_n <- nrow(data) - (nrow(data)*split)
           #Initialize class indices list
           class_indices <- list()
           for(class in names(output[["class_proportions"]])){
             #Get the indices with the corresponding categories
             indices <- which(data[,response_var] == class)
             #Add them to list
             class_indices[[class]] <-  indices
             #Store indices for training set
             output[["sample_indices"]][["training"]] <- c(output[["sample_indices"]][["training"]] ,sample(indices,size = round(training_n*output[["class_proportions"]][[class]],0), replace = F))
             #Remove indices to not add to test set
             indices <- indices[!(indices %in% output[["sample_indices"]][["training"]] )]
             #Add indices for test set
             output[["sample_indices"]][["test"]] <- c(output[["sample_indices"]][["test"]] ,sample(indices,size = round(test_n*output[["class_proportions"]][[class]],0), replace = F))
           }
           #Store proportions of data in training set
           output[["sample_proportions"]][["training"]] <- table(data[,response_var][output[["sample_indices"]][["training"]]])/sum(table(data[,response_var][output[["sample_indices"]][["training"]]]))
           #Store proportions of data  in test set
           output[["sample_proportions"]][["test"]] <- table(data[,response_var][output[["sample_indices"]][["test"]]])/sum(table(data[,response_var][output[["sample_indices"]][["test"]]]))
           output[["class_indices"]] <- class_indices
           #Output
           stratified.sampling_output <- list("output" = output)
         },
         "k-fold" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           #Get class indices
           class_indices <- output[["class_indices"]]
           #Initialize sample_indices for cv since it will be three levels
           output[["sample_indices"]][["cv"]] <- list()
           for(i in 1:k){
             #Keep initializing variable
             fold_idx <- c()
             output[["metrics"]][["cv"]][i,"Fold"] <- sprintf("Fold %s",i)
             #fold size; try to undershoot for excess
             fold_size <- floor(nrow(data)/k)
             for(class in names(output[["class_proportions"]])){
               fold_idx <- c(fold_idx, sample(class_indices[[class]],size = floor(fold_size*output[["class_proportions"]][[class]]), replace = F))
               #Remove already selected indices
               class_indices[[class]] <- class_indices[[class]][-which(class_indices[[class]] %in% fold_idx)]
             }
             #Add indices to list
             output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
             #Update proportions
             output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
           }
           #Deal with excess indices
           excess <- nrow(data) - length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
           if(excess > 0){
             for(class in names(output[["class_proportions"]])){
               fold_idx <- class_indices[[class]]
               if(length(fold_idx) > 0){
                 leftover <- rep(1:k,length(fold_idx))[1:length(fold_idx)]
                 for(i in 1:length(leftover)){
                   #Add indices to list
                   output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
                   #Update class proportions
                   output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
                 }
               }
             }
           }
           #Output
           stratified.sampling_output <- list("output" = output)
         }
  )
  }
