error.handling <- function(data = data, y.col = y.col,k = k,split = split, model.type = model.type, stratified = stratified, plot.metrics = plot.metrics){
  #Valid models
  valid.models <- c("lda","qda","logistic","svm","naivebayes")
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
  if(is.null(data)){
    stop("No input data")
  }
  if(length(y.col) != 1){
    stop("length of y.col must be 1")
  }
  if(is.numeric(y.col)){
    if(!(y.col %in% c(1:ncol(data)))){
      stop("y.col out of range")
    }

    } else if (is.character(y.col)){
      if(!(y.col %in% colnames(data))){
        stop("y.col not in dataframe")
      }
    } else{
      stop("y.col not a valid input")
    }
}
