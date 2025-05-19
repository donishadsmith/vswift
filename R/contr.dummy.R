.get_levels <- function(n, call) {
  if ((n_len <- length(n)) == 0 || n_len == 1 && !is.numeric(n) || n_len == 1 && is.numeric(n) && n < 2) {
    stop("'n' must specify more than one level")
  }
  
  return(if (n_len == 1 && is.numeric(n)) seq_len(n)
         else switch(call, "ordered" = levels(as.factor(n)), "unordered" = as.character(n)))
}

#' @export
contr.dummy <- function(n, contrasts=TRUE) {
  level_names <- .get_levels(n, call = "unordered")
  n_len <- length(level_names)
  mat <- matrix(0, nrow = n_len, ncol = n_len, dimnames = list(level_names, level_names))
  diag(mat) <- 1
  
  return(mat)
}


#' @export
contr.oridinal <- function(n, contrasts=TRUE) {
  level_names <- .get_levels(n, call = "ordered")
  n_len <- length(level_names)
  mat <- matrix(0.5, nrow = n_len, ncol = n_len - 1, dimnames = list(level_names, NULL))
  mat[lower.tri(mat)] <- -0.5
  
  return(mat)
}