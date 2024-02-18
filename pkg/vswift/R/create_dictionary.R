# Create dictionary for target variable if needed for certain algos
#' @noRd
#' @export
.create_dictionary <- function(preprocessed_data = NULL, target = NULL, classCV_output = NULL){
  counter <- 0
  new_classes <- c()
  for(class in names(table(preprocessed_data[,target]))){
    new_classes <- c(new_classes, paste(class, "=", counter, collapse = " "))
    classCV_output[["class_dictionary"]][[class]] <- counter
    counter <- counter + 1
  }
  if(!all(names(classCV_output[["class_dictionary"]]) == as.character(classCV_output[["class_dictionary"]]))){
    warning(sprintf("classes are now encoded: %s", paste(new_classes, collapse = ", ")))
  }
  return(classCV_output)
}