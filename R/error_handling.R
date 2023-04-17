.error.handling <- function(data = NULL, y_col = NULL,x_col = NULL,k = NULL,split = NULL, model_type = NULL, stratified = NULL,  random_seed = NULL,
                            call = NULL){
  #Valid models
  valid_models <- c("lda","qda","logistic","svm","naivebayes")
  if(all(!is.null(random_seed),!is.numeric(random_seed))){
    stop("random_seed must be a numerical scalar value")
  }
  # Ensure k is not an invalid number
  if(!is.data.frame(data)){
    stop("invalid input for data")
  }
  if(any(k %in% c(0,1), k < 0, k > 30,is.character(k), k != as.integer(k))){
    stop(sprintf("k = %s is not a valid input. `k` must be a non-negative integer between 2-30",k))
  }
  # Ensure split is between 0.5 to 0.8
  if(any(is.character(split), split < 0.5, split > 0.9)){
    stop("Input a split that is between 0.5 and 0.8")
  }
  # Ensure y and x matrices are valid
  if(is.null(data)){
    stop("No input data")
  }
  if(any(y_col == x_col, y_col %in% x_col)){
    stop("response variable cannot also be a predictor")
  }
  if(length(y_col) != 1){
    stop("length of y_col must be 1")
  }
  if(is.numeric(y_col)){
    if(!(y_col %in% c(1:ncol(data)))){
      stop("y_col out of range")
    }
    
  } else if (is.character(y_col)){
    if(!(y_col %in% colnames(data))){
      stop("y_col not in dataframe")
    }
  } else{
    stop("y_col must be an integer or character")
  }
  if(!is.null(x_col)){
    if(all(is.integer(x_col))){
      check_x <- 1:dim(data)[1]
    } else if(all(is.character(x_col))){
      check_x <- colnames(data)[colnames(data) != y_col]
    } else{
      stop("x_col must be a character vector or integer vector")
    }
    if(!(all(x_col %in% check_x))){
      stop("at least one predictor is not in dataframe")
    }
  }
  if(call == "categorical.cv.split"){
    # Ensure model_type has been assigned
    if(any(is.null(model_type), !(model_type %in% valid_models))){
      stop(sprintf("%s is an invalid model_type", model_type))
    }
    if(all(model_type == "logistic", length(levels(as.factor(data[,y_col]))) != 2)){
      stop("logistic regression requires a binary variable")
    }
  }
}