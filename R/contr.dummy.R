#' Internal Re-export of contr.dummy
#'
#' This function is re-exported from the \code{kknn} package to ensure compatibility with the \code{train.kknn} function,
#' which requires \code{contr.dummy} to be available in the current namespace. This function generates dummy variables
#' for unordered factors.
#'
#' @name contr.dummy
#' @keywords internal
#' @export
#' @importFrom kknn contr.dummy
NULL

#' Internal Re-export of contr.ordinal
#'
#' This function is re-exported from the \code{kknn} package to ensure compatibility with the \code{train.kknn} function,
#' which requires \code{contr.ordinal} to be available in the current namespace. This function generates dummy variables
#' for ordered factors.
#'
#' @name contr.ordinal
#' @keywords internal
#' @export
#' @importFrom kknn contr.ordinal
NULL