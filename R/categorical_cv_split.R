categorical.cv.split  <- function(data = NULL, y.col = NULL,k = NULL, split = 0.8, model.type = NULL, stratified = FALSE, plot.metrics = FALSE){
  " Parameters:
      -----------
      
     `data`: the dataframe containing the response variable and predictors
     `y.col`: the response variable to be analyzed
     `k`: a non-negative integer value less than 100 for k-fold cross validation
     `split`: a numeric value between 0.5 to 0.8 to determine the proportion of the dataset that will be used for training.
     `model.type`: The type of algorithm that will be used for data analysis. Currently, logistic regression, linear discriminant analysis (lda), and quadratic discriminant analysis (qda) are supported.
     `stratified`: If `TRUE`, stratefied sampling is used to maintain the relative proportion of the categories during data splitting.Default = NULL"
  
  # Checking if inputs are valid
  error.handling(data = data, y.col = y.col, k = k, split = split, model.type = model.type, stratified = stratified, plot.metrics = plot.metrics)
  #Assign 1 to iterator for future looping
  iterator <- 1
  if(is.character(y.col)){
    y.col <- which(colnames(data) == y.col)
  }
  #Remove rows with missing data
  cleaned_data <- data[complete.cases(data),]
  x <-  cleaned_data[,-y.col]
  y <- cleaned_data[,y.col]
  
  #Get category names
  #Convert y to numeric ranging starting with 0, while preserving original names in a dictionary
  categories.dict <- list()
  categories.length <- 0
  for(category in names(table(y))){
    if(is.numeric(y)){
      y[which(y == as.numeric(category))] <- categories.length
    } else{
      y[which(y == category)] <- categories.length
    }
    categories.dict[[as.character(categories.length)]] <- category
    categories.length <- categories.length + 1
  }
  y <- as.numeric(y)
  
  #Stratified sampling
  if(stratified == TRUE){
    #Get proportions
    categories.proportions <- table(y)/sum(table(y))
    #Split sizes
    training.n <- nrow(x)*split
    test.n <- nrow(x) - (nrow(x)*split)
    training.indices <- c()
    test.indices <- c()
    #Create category list that will be used for the cross validation loop
    categories.indices <- list()
    for(category in names(categories.dict)){
      #Get the indices with the corresponding categories
      indices <- which(y == as.numeric(category))
      #Add them to list
      categories.indices[[category]] <- indices
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
  set.metrics <- data.frame(matrix(nrow = 2, ncol = 1))
  colnames(set.metrics) <- "Set"
  set.metrics[1:2,"Set"] <- c("Training","Test")
  #Adding information to data frame
  if(!(is.null(k))){
    k.metrics <- data.frame(matrix(nrow = 1,ncol = 1))
    colnames(k.metrics) <- "Fold"
    #Create folds
    rows <- sample(1:nrow(x))
    #fold size
    fold.size <- nrow(x) %/% k
    #empty list
    folds <- list()
    #Creating non-overlapping folds while adding rownames to matrix
    for(i in 1:k){
      #Add name to dataframe
      k.metrics[i,"Fold"] <- sprintf("Fold %s",i)
      #Create fold with stratified or non stratified sampling
      if(stratified == TRUE){
        # Initialize variable
        fold.idx <- c()
        # fold.size changed to remaining rows left in the event nrow(x) is not divisible by k
        if(i == length(1:k)){
          fold.size <- nrow(x) - (fold.size*length(1:(k - 1)))
        }
        for(category in names(categories.dict)){
          if(i == length(1:k)){
            #Stratified sampling
            fold.idx <- c(fold.idx,categories.indices[[category]])
          } else{
            #Stratified sampling
            fold.idx <- c(fold.idx, sample(categories.indices[[category]],size = round(fold.size*categories.proportions[[category]],0), replace = F))
          }
          #Remove already selected indices
          categories.indices[[category]] <- categories.indices[[category]][-which(categories.indices[[category]] %in% fold.idx)]
        }
        #Add indices to list
        folds[[i]] <- fold.idx
      } else{
        if(i == length(1:k)) {
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
           "logistic" = {model <- glm(Y.data ~ X.data, family = "binomial")},
           "svm" = {model <- e1071::svm(as.factor(Y.data) ~ X.data)},
           "naivebayes" = {model <- naivebayes::naive_bayes(x = X.data, y= as.factor(Y.data))}
    )
    # Perform classification accuracy for training and test data split
    if(i == 1){
      for(j in c("Training","Test")){
        if(j == "Test"){
          #Assign test set to new variable
          X.data <- X.val
          Y.data <- Y.val
        }
        # Get prediction
        switch(model.type,
               "svm" = {prediction.vector <- as.numeric(predict(model, newdata = data.frame(X.data))) - 1},
               "logistic" = {prediction.vector <- ifelse(prediction.vector > 0.5, 1, 0)},
               "naivebayes" = {prediction.vector <- as.numeric(predict(model, newdata = data.frame(X.data)))},
               prediction.vector <- as.numeric(predict(model, newx = X.data)$class) - 1
        )
        set.metrics[which(set.metrics$Set == j),"Classification Accuracy"] <- sum(Y.data == prediction.vector)/length(Y.data)
        for(category in names(categories.dict)){
          #Sum of category that was correctly guessed
          true.pos <- sum(Y.data[which(Y.data == as.numeric(category))] == prediction.vector[which(Y.data  == as.numeric(category))])
          #Sum of the missed category
          false.neg <- abs(true.pos  - length(Y.data[which(Y.data  == as.numeric(category))]))
          #Sum of the classified category; Taking "One Class vs. Rest" approach
          false.pos <- length(which(prediction.vector[-which(Y.data  == as.numeric(category))] == as.numeric(category)))
          set.metrics[which(set.metrics$Set == j),sprintf("Class %s Precision", categories.dict[category])] <- true.pos/(true.pos + false.pos)
          set.metrics[which(set.metrics$Set == j),sprintf("Class %s Recall", categories.dict[category])] <- true.pos/(true.pos + false.neg)
          set.metrics[which(set.metrics$Set == j),sprintf("Class %s F1", categories.dict[category])] <- 2/(1/(true.pos/(true.pos + false.pos)) + 1/(true.pos/(true.pos + false.neg)))
        }
      }
    } else{
      if(all(!(is.null(k)),(i-1) <= k)){
        #Assign validation set to new variables
        X.data <- X.val
        Y.data <- Y.val
        
      
        # Get prediction
        switch(model.type,
               "svm" = {prediction.vector <- as.numeric(predict(model, newdata = data.frame(X.data))) - 1},
               "logistic" = {prediction.vector <- ifelse(prediction.vector > 0.5, 1, 0)},
               "naivebayes" = {prediction.vector <- as.numeric(predict(model, newdata = data.frame(X.data)))},
               prediction.vector <- as.numeric(predict(model, newx = X.data)$class) - 1
        )
        #Calculate classification accuracy for fold
        k.metrics[which(k.metrics$Fold == sprintf("Fold %s",i-1)), "Classification Accuracy"] <- sum(Y.data == prediction.vector)/nrow(Y.data)
        for(category in names(categories.dict)){
          #Sum of category that was correctly guessed
          true.pos <- sum(Y.data[which(Y.data == as.numeric(category))] == prediction.vector[which(Y.data  == as.numeric(category))])
          #Sum of the missed category
          false.neg <- abs(true.pos  - length(Y.data[which(Y.data  == as.numeric(category))]))
          #Sum of the classified category
          false.pos <- length(which(prediction.vector[-which(Y.data  == as.numeric(category))] == as.numeric(category)))
          k.metrics[which(k.metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Class %s Precision", categories.dict[category])] <- true.pos/(true.pos + false.pos)
          k.metrics[which(k.metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Class %s Recall", categories.dict[category])] <- true.pos/(true.pos + false.neg)
          k.metrics[which(k.metrics$Fold == sprintf("Fold %s",i-1)), sprintf("Class %s F1", categories.dict[category])] <- 2/(1/(true.pos/(true.pos + false.pos)) + 1/(true.pos/(true.pos + false.neg)))
        }
        #Calculate final metrics and plot
      }
      if(all(!(is.null(k)),(i-1) == k)){
        # To get the correct category for plot title
        category.idx <- 1
        #Get the last row index
        idx <- nrow(k.metrics)
        #Initialize new metrics
        k.metrics[(idx + 1):(idx + 3),"Fold"] <- c("Mean CV:","Standard Deviation CV:","Standard Error CV:")
        # Calculate mean, standard deviation, and sd for each column except for fold
        for(colname in colnames(k.metrics)[colnames(k.metrics) != "Fold"]){
          # Create vector containing corresponding column name values for each fold
          num.vector <- k.metrics[1:idx, colname]
          k.metrics[which(k.metrics$Fold == "Mean CV:"),colname] <- mean(num.vector)
          k.metrics[which(k.metrics$Fold == "Standard Deviation CV:"),colname] <- sd(num.vector)
          k.metrics[which(k.metrics$Fold == "Standard Error CV:"),colname] <- sd(num.vector)/sqrt(k)
          #Split column name
          split.vector <- unlist(strsplit(colname, split = " "))
          #Plot metrics
          if(plot.metrics == TRUE){
            # depending on column name, plotting is handled slightly differently
            if("Classification" %in% split.vector){
              plot(x = 1:k, y = num.vector, ylim = c(0,1), xlab = "K-folds", ylab = "Classification Accuracy" , xaxt = "n")
              axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
            } else{
              # Get correct metric name for plot y title
              y.name <- c("Precision","Recall","F1")[which(c("Precision","Recall","F1") %in% split.vector)]
              
              plot(x = 1:k, y = num.vector, ylim = c(0,1), xlab = "K-folds", ylab = y.name, main = paste("Category: ",categories.dict[[category.idx]]), xaxt = "n")
              axis(side = 1, at = as.integer(1:k), labels = as.integer(1:k))
              # Add 1 to `category.idx` when `y.name == "Recall"` to get correct category plot title
              if(y.name == "F1"){
                category.idx <- category.idx + 1
              }
            }
            # Add mean and standard deviation to the plot
            abline(h = mean(num.vector), col = "red", lwd = 1)
            abline(h = mean(num.vector) + sd(num.vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
            abline(h = mean(num.vector) - sd(num.vector)/sqrt(k), col = "blue", lty = 2, lwd = 1)
          }
        }
      }
    }
  }
  # Create list of output depending on if k validation is done or not
  if(is.null(k)){
    output <- list('data.split' = set.metrics)
    } else{
      output <- list('data.split' = set.metrics,"k-fold" = k.metrics)
      }
  return(output)
  }


