# Code for checking the typing of parameters
.type_validator <- function(param, value) {
  msg <- "`%s` must be of the following classes: %s"
  types <- .PARAM_TYPES$primary[[param]]

  if ("list" %in% types) {
    secondary_names <- names(value)
    for (name in secondary_names) {
      types <- .PARAM_TYPES$secondary[[name]]
      if (!inherits(value[[name]], types)) stop(sprintf(msg, name, paste(types, collapse = ", ")))
    }
  } else {
    if (!inherits(value, types)) stop(sprintf(msg, param, paste(types, collapse = ", ")))
  }
}
