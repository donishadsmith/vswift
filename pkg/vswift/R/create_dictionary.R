# Create dictionary for target variable if needed for certain algos
.create_dictionary <- function(target_vector) {
  counter <- 0
  new_classes <- c()
  class_dict <- list()

  for (class in names(table(target_vector))) {
    new_classes <- c(new_classes, paste(class, "=", counter, collapse = " "))
    class_dict[[class]] <- counter
    counter <- counter + 1
  }

  warning(sprintf(
    "creating keys for target variable due to 'logistic' or 'gbm' being specified;\n  classes are now encoded: %s",
    paste(new_classes, collapse = ", ")
  ))

  return(class_dict)
}
