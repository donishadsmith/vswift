#Cross validation function for generalization
categorical.cv.split  <- function(x = NULL, y = NULL, k = NULL, split = 0.8, model.type = NULL, stratified = NULL){
  " Parameters:
      -----------
      
     `x`: predictor matrix
     `y`: response variable matrix
     `k`: a non-negative integer value less than 100 for k-fold cross validation
     `split`: a numeric value between 0.5 to 0.8 to determine the proportion of the dataset that will be used for training.
     `model.type`: The type of algorithm that will be used for data analysis. Currently, logistic regression, linear discriminant analysis (lda), and quadratic discriminant analysis (qda) are supported.
     `stratified`: If `TRUE`, stratefied sampling is used to maintain the relative proportion of the categories during data splitting.Default = NULL"
  
  valid.models <- c("lda","qda","logistic")
  # Checking if inputs are valid
  
  # Ensure k is not an invalid number
  if(any(k == 0, k < 0, k > 99,is.character(k), k != as.integer(k))){
    stop(sprintf("k = %s is not a valid input",k))
  }
  # Ensure model.type has been assigned
  if(any(is.null(model.type), !(model.type %in% valid.models))){
    stop(sprintf("%s is an invalid model.type", model.type))
  }
  # Ensure split is between 0.5 to 0.8
  if(any(is.character(split), split < 0.5, split > 0.8)){
    stop("Input a split that is between 0.5 and 0.8")
  }
  # Ensure y and x matrices are valid
  if(any(is.null(x), is.null(y), dim(as.matrix(y)[2] != 1))){
    if(dim(as.matrix(y))[2] != 1){
      stop("More than one column in y matrix")
    }
    else{
      stop("Input a valid x and y matrix")
    }
  }
  #Ensure y matrix is categorical
  if(!(all(is.numeric(y), all(y == as.integer(y))))){
    stop("y matrix needs to a categorical vector")
  }
  #Ensure y matrix as no more or less than two levels if a logistic model is chosen
  if(all(model.type == "logistic",length(levels(as.factor(y))) != 2 )){
    stop("y matrix must have two levels for logistic models")
    
  }
  # Ensure x and y matrices have the same number of rows
  if(dim(y)[1] != dim(x)[1]){
    stop("x and y matrices have unequal rows")
  } 
  
  #Assign 1 to iterator for future looping
  iterator <- 1
  
  #Remove rows with missing data
  data <- cbind(x,y)
  cleaned_data <- data[complete.cases(data),]
  x <-  cleaned_data[,-ncol(cleaned_data)]
  y <- cleaned_data[,ncol(cleaned_data)]
  
  #Stratified sampling
  if(stratified == TRUE){
    #Get category names
    categories <- names(table(y))
    #Get proportions
    categories.proportions <- table(y)/sum(table(y))
    #Split sizes
    training.n <- nrow(x)*split
    test.n <- nrow(x) - (nrow(x)*split)
    training.indices <- c()
    test.indices <- c()
    #Create category list that will be used for the cross validation loop
    category.list <- list()
    for(category in categories){
      #Get the indices with the corresponding categories
      indices <- which(y == as.numeric(category))
      #Add them to list
      category.list[[category]] <- indices
      training.indices <- c(training.indices,sample(indices,size = round(training.n*categories.proportions[[category]],0), replace = F))
      #Remove indices to not add to test set
      indices <- indices[!(indices %in% training.indices)]
      
      test.indices <- c(test.indices,sample(indices,size = round(test.n*categories.proportions[[category]],0), replace = F))
    }

    X.train <- as.matrix(x[training.indices,]); Y.train <- as.matrix(y[training.indices])
    X.val <- as.matrix(x[test.indices,]); Y.val <- as.matrix(y[test.indices])

  } else{
    #Data split
    training.indices <- sample(1:nrow(x),size = round(nrow(x)*split,0),replace = F)
    X.train <- as.matrix(x[training.indices,]);  Y.train <- as.matrix(y[training.indices])
    X.val <- as.matrix(x[-training.indices,]); Y.val <- as.matrix(y[-training.indices])
  }
  
  #Create data table
  evaluation.metrics <- data.frame("Training" = NA, "Test" = NA)
  rownames(evaluation.metrics) <- "Classification Accuracy"
  for(category in categories){
    evaluation.metrics[(nrow(evaluation.metrics) + 1):(nrow(evaluation.metrics) + 2),] <- NA
    rownames(evaluation.metrics)[(length(rownames(evaluation.metrics)) - 1):length(rownames(evaluation.metrics))] <- c(sprintf("Class %s Precision", category),sprintf("Class %s Recall", category))
  }
  #Adding information to data frame
  if(!(is.null(k))){
    evaluation.metrics$`K-fold CV` <- NA
    #Create folds
    rows <- sample(1:nrow(x))
    #fold size
    fold.size <- nrow(x) %/% k
    #empty list
    folds <- list()
    final.metrics.names.vector <- c()
    #Creating non-overlapping folds while adding rownames to matrix
    for(i in 1:k){
      evaluation.metrics[nrow(evaluation.metrics):(nrow(evaluation.metrics) + 1),] <- NA
      rownames(evaluation.metrics)[nrow(evaluation.metrics)] <- sprintf("Fold %s Classification Accuracy", i)
      for(category in categories){
        evaluation.metrics[nrow(evaluation.metrics):(nrow(evaluation.metrics) + 2),] <- NA
        rownames(evaluation.metrics)[(nrow(evaluation.metrics) - 1):nrow(evaluation.metrics)] <- c(sprintf("Fold %s Class %s Precision", i,category),sprintf("Fold %s Class %s Recall", i,category))
        # Names for final metrics
        if(i == k){
          final.metrics.names.vector <- c(final.metrics.names.vector,sprintf("Mean CV Class %s Precision", category),sprintf("SD CV Class %s Precision", category),
                                           sprintf("Mean CV Class %s Recall", category), sprintf("SD CV Class %s Recall", category))
          
        }
      }
      #Create fold with stratified or non stratified sampling
      if(stratified == TRUE){
        # Initialize variable
        fold.idx <- c()
        # fold.size changed to remaining rows left in the event nrow(x) is not divisible by k
        if(i == length(1:k)){
          fold.size <- nrow(x) - (fold.size*length(1:(k - 1)))
        }
        for(category in categories){
          if(i == length(1:k)){
            #Stratified sampling
            fold.idx <- c(fold.idx,category.list[[category]])
          } else{
            #Stratified sampling
            fold.idx <- c(fold.idx, sample(category.list[[category]],size = round(fold.size*categories.proportions[[category]],0), replace = F))
          }
          #Remove already selected indices
          category.list[[category]] <- category.list[[category]][-which(category.list[[category]] %in% fold.idx)]
        }
        #Add indices to list
        folds[[i]] <- fold.idx
        } else{
        if (i == length(1:k)) {
          #Folds may not have equal sizes
          fold.idx <- rows
          } else {
          fold.idx <- rows[1:fold.size]
          #Remove rows from vectors to prevent overlapping,last fold may be smaller or larger than other folds
          rows <- rows[-c(1:fold.size)]
          }
        #Add indices to list
        folds[[i]] <- fold.idx
        }
      }
    #Add additional rownames to store corresponding values
    evaluation.metrics[nrow(evaluation.metrics):(nrow(evaluation.metrics) + length(final.metrics.names.vector) + 3),] <- NA
    #Get character vector of numbers equal to the number of rows to select rownames that need to be changed
    num.vector <- as.character(1:length(rownames(evaluation.metrics)))
    #Used later to calculate metrics
    indices <- which(rownames(evaluation.metrics) %in% num.vector)
    rownames(evaluation.metrics)[indices] <- c("Mean & SD","Mean CV Classification Accuracy", "SD CV Classification Accuracy", final.metrics.names.vector)
    #Assign k to iterator
    iterator <- k
    }
  #Ensure model type is lowercase
  model.type <- tolower(model.type)
  #Add it plus one to the iterator if k is not null
  iterator <- ifelse(!(is.null(k)), iterator + 1, iterator)
  #First iteration will always be the evaluation for the traditional data split method
  for(i in 1:iterator){
    if(i == 1){
      # Assigning data split matrices to new variable 
      X.data <- X.train
      Y.data <- Y.train
      } else{
        # After the first iteration the cv begins, the training set is assigned to a new variable
        X.data <- as.matrix(x[-c(folds[[i-1]]), ]) ; Y.data <- as.matrix(y[-c(folds[[i-1]])])
        X.val <- as.matrix(x[c(folds[[i-1]]),]) ; Y.val <- as.matrix(y[c(folds[[i-1]])])
        }
    #Generate model depending on chosen model.type
    switch(model.type,
           "lda" = {model <- MASS::lda(as.factor(Y.data) ~ X.data)},
           "qda" = {model <- MASS::qda(as.factor(Y.data) ~ X.data)},
           "logistic" = {model <- glm(Y.data ~ X.data, family = "binomial")}
           )
    # Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("Training","Test")){
        if(j == "Training"){
          #Obtain prediction vector
          if(model.type != "logistic"){
            prediction.vector <- as.numeric(predict(model, newx = X.data)$class)
            #Subtract prediction vector if 0 is a level in y
            if("0" %in% levels(as.factor(y))){
              prediction.vector <- prediction.vector - 1
              }
            } else{
              prediction.vector <- predict(model, newdata = data.frame(X.data), type = "response")
              prediction.vector <- ifelse(prediction.vector > 0.5, 1, 0)
            }
          
          evaluation.metrics["Classification Accuracy", j] <- sum(Y.train == prediction.vector)/length(Y.train)
          for(category in categories){
            #Sum of category that was correctly guessed
            true.pos <- sum(Y.data[which(Y.data == as.numeric(category))] == prediction.vector[which(Y.data  == as.numeric(category))])
            #Sum of the missed category
            false.neg <- abs(true.pos  - length(Y.data[which(Y.data  == as.numeric(category))]))
            #Sum of the classified category
            false.pos <- length(which(prediction.vector[-which(Y.data  == as.numeric(category))] == as.numeric(category)))
            evaluation.metrics[sprintf("Class %s Precision", category), j] <- true.pos/(true.pos + false.pos)
            evaluation.metrics[sprintf("Class %s Recall", category), j] <- true.pos/(true.pos + false.neg)
            }
          } else{
          #Assign test set to new variable
          X.data <- X.val
          Y.data <- Y.val
          if(model.type != "logistic"){
            #Obtain prediction vector
            prediction.vector <- as.numeric(predict(model, newx = X.data)$class)
            #Subtract prediction vector if 0 is a level in y
            if("0" %in% levels(as.factor(y))){
              prediction.vector <- prediction.vector - 1
              }
            }else {
              prediction.vector <- predict(model, newdata = data.frame(X.data), type = "response")
              prediction.vector <- ifelse(prediction.vector > 0.5, 1, 0)
            }
          evaluation.metrics["Classification Accuracy", j] <- sum(Y.data == prediction.vector)/length(Y.data)
          for(category in categories){
            #Sum of category that was correctly guessed
            true.pos <- sum(Y.data[which(Y.data == as.numeric(category))] == prediction.vector[which(Y.data  == as.numeric(category))])
            #Sum of the missed category
            false.neg <- abs(true.pos  - length(Y.data[which(Y.data  == as.numeric(category))]))
            #Sum of the classified category
            false.pos <- length(which(prediction.vector[-which(Y.data  == as.numeric(category))] == as.numeric(category)))
            evaluation.metrics[sprintf("Class %s Precision", category), j] <- true.pos/(true.pos + false.pos)
            evaluation.metrics[sprintf("Class %s Recall", category), j] <- true.pos/(true.pos + false.neg)
          }
          }
        }
      } else{
          if(!(is.null(k))){
            if((i-1) <= k){
              #Assign validation set to new variables
              X.data <- X.val
              Y.data <- Y.val
              if(model.type != "logistic"){
                #Obtain prediction vector
                prediction.vector <- as.numeric(predict(model, newx = X.data)$class)
                #Subtract prediction vector if 0 is a level in y
                if("0" %in% levels(as.factor(y))){
                  prediction.vector <- prediction.vector - 1
                  }
                }else{
                  prediction.vector <- predict(model, newdata = data.frame(X.data), type = "response")
                  prediction.vector <- ifelse(prediction.vector > 0.5, 1, 0)
                }
              
              #Calculate classification accuracy for fold
              evaluation.metrics[sprintf("Fold %s Classification Accuracy",i-1), "K-fold CV"] <- sum(Y.data == prediction.vector)/nrow(Y.data)
              for(category in categories){
                #Sum of category that was correctly guessed
                true.pos <- sum(Y.data[which(Y.data == as.numeric(category))] == prediction.vector[which(Y.data  == as.numeric(category))])
                #Sum of the missed category
                false.neg <- abs(true.pos  - length(Y.data[which(Y.data  == as.numeric(category))]))
                #Sum of the classified category
                false.pos <- length(which(prediction.vector[-which(Y.data  == as.numeric(category))] == as.numeric(category)))
                evaluation.metrics[sprintf("Fold %s Class %s Precision",i - 1, category), "K-fold CV"] <- true.pos/(true.pos + false.pos)
                evaluation.metrics[sprintf("Fold %s Class %s Recall", i - 1,category), "K-fold CV"] <- true.pos/(true.pos + false.neg)
                }
              #Calculate final metrics for CV
              #Work in progress
              if((i-1) == k){
                #Add for loop to calculate these metrics'
                #Get the relevant names
                names.vectors <- (rownames(evaluation.metrics))[(2 + length(categories)*2):(nrow(evaluation.metrics)-length(indices))]
                split.vector <- c()
                for(metric in c("Classification Accuracy","Precision", "Recall")){
                  for(category in categories){
                    for(idx in 1:length(names.vectors)){
                      # Splitting row name
                      split.name <- unlist(strsplit(names.vectors[idx], split = " "))
                      # Split differently depending on metric being calculated
                      if(metric != "Classification Accuracy"){
                        split.name <- split.name[3:length(split.name)]
                        } 
                      #If metric name and category in split.name, add to vector
                      if(as.numeric(category) < 10){
                        switch(metric,
                               "Classification Accuracy" = {
                                 if(all("Classification" %in% split.name)){
                                   split.vector <- c(split.vector,names.vectors[idx])
                                   }},
                               if(all(metric %in% split.name, category %in% split.name)){
                                 split.vector <- c(split.vector,names.vectors[idx])
                               })
                        }else{
                          num.vec <- unlist(strsplit(category, split = NULL))
                          switch(metric,
                                 "Classification Accuracy" = {
                                   if(all("Classification" %in% split.name)){
                                     split.vector <- c(split.vector,names.vectors[idx])
                                   }},
                                 if(all(metric %in% split.name, num.vec[1] %in% split.name & num.vec[2] %in% split.name)){
                                   split.vector <- c(split.vector,names.vectors[idx])
                                 })
                          }
                      if(idx == length(names.vectors)){
                        if(metric == "Classification Accuracy"){
                          evaluation.metrics[sprintf("Mean CV %s", metric), "K-fold CV"] <- mean(evaluation.metrics[split.vector, "K-fold CV"])
                          evaluation.metrics[sprintf("SD CV %s", metric), "K-fold CV"] <- sd(evaluation.metrics[split.vector, "K-fold CV"])
                          }else{
                            evaluation.metrics[sprintf("Mean CV Class %s %s", category, metric), "K-fold CV"] <- mean(evaluation.metrics[split.vector, "K-fold CV"])
                            evaluation.metrics[sprintf("SD CV Class %s %s", category, metric), "K-fold CV"] <- sd(evaluation.metrics[split.vector, "K-fold CV"])
                            }
                        #Empty vector
                        split.vector <- c()
                      }
                    }
                  }
                }
              }
            }
          }
      }
    }
  return(evaluation.metrics)
  }


