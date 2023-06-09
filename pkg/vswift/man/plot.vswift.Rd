% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.vswift.R
\name{plot.vswift}
\alias{plot.vswift}
\title{Plot model evaluation metrics}
\usage{
\method{plot}{vswift}(
  object,
  split = TRUE,
  cv = TRUE,
  save_plots = FALSE,
  path = NULL,
  model_type = NULL,
  ...
)
}
\arguments{
\item{object}{An object of class vswift.}

\item{split}{A logical value indicating whether to plot metrics for train-test splitting results. Default = TRUE.}

\item{cv}{A logical value indicating whether to plot metrics for k-fold cross-validation results. Note: Solid red line represents the mean
and dashed blue line represents the standard deviation. Default = TRUE.}

\item{save_plots}{A logical value to save all plots as separate png files. Plot will not be displayed if set to TRUE. Default = FALSE.}

\item{path}{A character representing the file location, with trailing slash, to save to. If not specified, the plots will be saved to the current
working directory.}

\item{model_type}{A character or vector of the model metrics to be printed. If not specified, all model metrics will be printed. Available options:
"lda" (Linear Discriminant Analysis), "qda" (Quadratic Discriminant Analysis), 
"logistic" (Logistic Regression), "svm" (Support Vector Machines), "naivebayes" (Naive Bayes), 
"ann" (Artificial Neural Network), "knn" (K-Nearest Neighbors), "decisiontree" (Decision Tree), 
"randomforest" (Random Forest), "multinom" (Multinomial Logistic Regression), "gbm" (Gradient Boosting Machine).}

\item{...}{Additional arguments that can be passed to the `png()` function.}
}
\value{
Plots representing evaluation metrics.
}
\description{
`plot.vswift` plots model evaluation metrics (classification accuracy and precision, recall, and f-score for each class) from a vswift object.
}
\examples{
# Load an example dataset

data(iris)

# Perform a train-test split with an 80\% training set and stratified_sampling using QDA

result <- classCV(data = iris, target = "Species", split = 0.8,
model_type = "qda", stratified = TRUE)

# Plot performance metrics for train-test split

plot(result)

}
\author{
Donisha Smith
}
