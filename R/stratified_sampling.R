stratified.sampling <- function(type, output, data, response_var, split = NULL, fold_size = NULL, k_metrics = NULL, k = NULL,
                                class_indices = NULL, random_seed = NULL){
  switch(type,
         "split" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           #Get proportions
           categories_proportions <- table(data[,response_var])/sum(table(data[,response_var]))
           #Split sizes
           training_n <- nrow(data)*split
           test_n <- nrow(data) - (nrow(data)*split)
           training_indices <- c()
           test_indices <- c()
           #Create category list that will be used for the cross validation loop
           class_indices <- list()
           output[["class_indices"]] <- list()
           for(category in names(output[["class_dict"]])){
             #Get the indices with the corresponding categories
             indices <- which(data[,response_var] == as.numeric(category))
             #Add them to list
             output[["class_indices"]][[category]] <- class_indices[[category]] <- indices
             training_indices <- c(training_indices,sample(indices,size = round(training_n*categories_proportions[[category]],0), replace = F))
             
             #Remove indices to not add to test set
             indices <- indices[!(indices %in% training_indices)]
             test_indices <- c(test_indices,sample(indices,size = round(test_n*categories_proportions[[category]],0), replace = F))
           }
           # Update output
           output[["sample_indices"]][["training"]] <- training_indices
           output[["sample_indices"]][["test"]] <- test_indices
           output[["sample_proportions"]] <- list()
           # Store proportions of data and change list names
           output[["sample_proportions"]][["training"]] <- table(data[,response_var][training_indices])/sum(table(data[,response_var][training_indices]))
           names(output[["sample_proportions"]][["training"]]) <- as.character(output[["class_dict"]])
           # Store proportions of data and change list names
           output[["sample_proportions"]][["test"]] <- table(data[,response_var][test_indices])/sum(table(data[,response_var][test_indices]))
           names(output[["sample_proportions"]][["test"]]) <- as.character(output[["class_dict"]])
           names(output[["class_indices"]]) <-  as.vector(output[["class_dict"]])
           
           stratified.sampling_output <- list("training" = training_indices, "test" = test_indices, "output" = output, "class_indices" = class_indices)
         },
         "k-fold" = {
           #Set seed
           if(!is.null(random_seed)){
             set.seed(random_seed)
           }
           categories_proportions <- table(data[,response_var])/sum(table(data[,response_var]))
           for(i in 1:k){
             # Keep Initialize variable
             fold_idx <- c()
             k_metrics[i,"Fold"] <- sprintf("Fold %s",i)
             #fold size; try to undershoot for excess
             fold_size <- floor(nrow(data)/k)
             for(category in names(output[["class_dict"]])){
               fold_idx <- c(fold_idx, sample(class_indices[[category]],size = floor(fold_size*categories_proportions[[category]]), replace = F))
               #Remove already selected indices
               class_indices[[category]] <- class_indices[[category]][-which(class_indices[[category]] %in% fold_idx)]
             }
             #Add indices to list
             output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]] <- fold_idx
             # Update proportions
             output[["sample_proportions"]][["cv"]][[sprintf("fold %s",i)]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",i)]]]))
           }
           #Deal with excess
           
           excess <- nrow(data)- length(as.numeric(unlist(output[["sample_indices"]][["cv"]])))
           if(excess > 0){
             for(category in names(output[["class_dict"]])){
               fold_idx <- class_indices[[category]]
               if(length(fold_idx) > 0){
                 leftover <- rep(1:k,length(fold_idx))[1:length(fold_idx)]
                 for(i in 1:length(leftover)){
                   #Add indices to list
                   output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- c(fold_idx[i],output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]])
                   # Get proportions
                   output[["sample_proportions"]][["cv"]][[sprintf("fold %s",leftover[i])]] <- table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]])/sum(table(data[,response_var][output[["sample_indices"]][["cv"]][[sprintf("fold %s",leftover[i])]]]))
                 }
               }
             }
           }
           stratified.sampling_output <- list("output" = output, "k_metrics" = k_metrics)
         }
  )
}


