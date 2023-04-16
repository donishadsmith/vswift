continuous.cv.split <- function(X,Y,K,s.range,model_type, random_seed){
  set.seed(123)
  #Check to see if matrices match
  if(nrow(X) != nrow(Y)){
    print('length of X and Y matrix do not match.')
  } else{
    #Create matrix table
    table.mat <- matrix(nrow = (K + 4), ncol = length(s.range))
    colnames(table.mat) <- as.character(s.range)
    row_names <- c()
    #Prepare for splitting
    n <- nrow(X)
    #Shuffle rows for randomization in folds
    rows <- sample(1:n)
    #fold size
    fold.size <- n %/% K
    #empty list
    folds <- list()
    # Creating folds with random indices and adding column names and rownames to matrix
    for(i in 1:K){
      #Create row for each fold MSE that will be calculated
      row_names <- c(row_names, sprintf("Fold %s MSE", i))
      #Create fold
      if (i == length(1:K)) {
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
    row.names(table.mat) <- c(row_names, "Mean CV MSE","SD CV MSE","SE CV MSE","One Standard Rule")
    
    # Conduct validation using each fold
    
    for(i in 1:K){
      
      
      X.train <- X[-c(folds[[i]]), ] ; Y.train <- Y[-c(folds[[i]]),]
      X.val <- X[c(folds[[i]]),] ; Y.val <- Y[c(folds[[i]]),]
      #Generate model
      model <- lars(x = X.train, y = Y.train, use.Gram = F, type = mod, normalize = T, intercept = T)
      
      for(j in s.range){
        # Use each tuning parameter for prediction
        table.mat[i,as.character(j)] <- mean((Y.val - predict(model,type = "fit", newx = X.val, s = j, mode ='fraction')$fit)^2)
        #Calculate Mean CV MSE
        table.mat["Mean CV MSE",as.character(j)] <- mean(table.mat[1:3,as.character(j)])
        #Calculate SD CV MSE
        table.mat["SD CV MSE",as.character(j)] <- sd(table.mat[1:3,as.character(j)])
        #Calculate Standard error
        table.mat["SE CV MSE",as.character(j)] <- table.mat[5,as.character(j)]/(sqrt(K))
        #Apply one-standard rule'
        if(j == s.range[length(s.range)]){
          best.idx <- which.min(table.mat["Mean CV MSE",1:ncol(table.mat)])
          threshold <- table.mat["Mean CV MSE",1:ncol(table.mat)][best.idx] + table.mat["SE CV MSE",1:ncol(table.mat)][best.idx]
          cutoff <- c()
          for(s in s.range){
            if(all(table.mat["Mean CV MSE",as.character(s)] < threshold, s < s.range[best.idx])){
              cutoff <- c(cutoff, which(s.range == s))
            }
          }
          optimal.s <- s.range[cutoff[which.max(table.mat["Mean CV MSE",cutoff])]]
          table.mat["One Standard Rule",cutoff[which.max(table.mat["Mean CV MSE",cutoff])]] <- optimal.s
          #Convert to dataframe to use flextable package
          table.df <- cbind(row.names(table.mat),as.data.frame(table.mat))
          colnames(table.df) <- c("S",colnames(table.df)[2:length(colnames(table.df))])
          outputs = list("table.mat" = table.mat, "table.df"= table.df)
        }
        
      }
    }
  }
  
  return(outputs)
}