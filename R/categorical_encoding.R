# Create dictionary for target variable if needed for certain algos
.create_dictionary <- function(target_vector, alternate_warning = FALSE) {
  counter <- 0
  new_classes <- c()
  class_dict <- list()

  for (class in names(table(target_vector))) {
    new_classes <- c(new_classes, paste(class, "=", counter, collapse = " "))
    class_dict[[class]] <- counter
    counter <- counter + 1
  }

  msg <- if (!alternate_warning) "due to 'logistic' or 'xgboost' being specified" else "for `rocCurve`"

  warning(sprintf(
    "creating keys for target variable %s;\n  classes are now encoded: %s",
    msg, paste(new_classes, collapse = ", ")
  ))

  return(class_dict)
}

# Helper function to convert keys
.convert_keys <- function(target_vector, keys, direction) {
  if (direction == "encode") {
    labels <- sapply(target_vector, function(x) keys[[x]])
  } else {
    converted_keys <- as.list(names(keys))
    names(converted_keys) <- as.character(as.vector(unlist(keys)))
    labels <- sapply(target_vector, function(x) converted_keys[[as.character(x)]])
  }
  return(labels)
}
