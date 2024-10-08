% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/genFolds.R
\name{genFolds}
\alias{genFolds}
\title{Create split datasets and/or folds with optional stratification}
\usage{
genFolds(
  data,
  target,
  train_params = list(split = NULL, n_folds = NULL, stratified = FALSE, random_seed =
    NULL),
  create_data = FALSE
)
}
\arguments{
\item{data}{A data frame.}

\item{target}{A numerical index or character name for the target variable. Only needs to be specified if
\code{stratified = TRUE}. Default = \code{NULL}.}

\item{train_params}{A list that can contain the following parameters:
\itemize{
\item \code{"split"}: A number from 0 to 1 for the proportion of data to use for the
training set, leaving the rest for the test set. If not specified, train-test splitting will not
be done. Note, this parameter is used to perform train-test splitting, which is separate
from cross-validation. Can be set to NULL, to not perform train-test splitting.
Default = \code{NULL}.
\item \code{"n_folds"}: An integer greater than 2 that indicates the number of folds to use for
k-fold cross validation (CV). Note, k-fold CV is performed separately from train-test splitting.
Can be set to NULL, to not perform k-fold CV. Default = \code{NULL}
\item \code{"stratified"}: A logical value indicating if stratified sampling should be used.
Default = \code{FALSE}.
\item \code{"random_seed"} A numerical value for the random seed to ensure random splitting
are reproducible. Default = \code{NULL}.
}}

\item{create_data}{A logical value indicating whether to create all training and test/validation data frames.
Default = \code{FALSE}.}
}
\value{
A list containing the indices for train-test splitting and/or k-fold cross-validation, with information on
        the class distribution in the training, test sets, and folds (if applicable) as well as the generated split
        datasets and folds based on the indices.
}
\description{
Standalone function generates train-test split datasets and/or k-fold cross-validation folds, with the
             option to perform stratified sampling based on class distribution.
}
\examples{
# Load example dataset

data(iris)

# Obtain indices for 80\% training/test split and 5-fold CV

output <- genFolds(data = iris, target = "Species", train_params = list(split = 0.8, n_folds = 5))

}
\author{
Donisha Smith
}
