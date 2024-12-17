# Function to partition data
.partition <- function(data, subsets) {
  df_list <- list()
  # Get data for train-test split
  if ("split" %in% names(subsets)) {
    df_list$split$train <- data[subsets$split$train, ]
    df_list$split$test <- data[subsets$split$test, ]
  }

  # Get train and test partitions for cv
  if ("cv" %in% names(subsets)) {
    for (fold in names(subsets$cv)) {
      df_list$cv[[fold]]$train <- data[-subsets$cv[[fold]], ]
      df_list$cv[[fold]]$test <- data[subsets$cv[[fold]], ]
    }
  }

  return(df_list)
}
