# Function to perform train-test split with no stratification
.split <- function(N, split, random_seed) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)

  split_indxs <- list("train" = NULL, "test" = NULL)
  # Create test and train set
  train_indxs <- sample(1:N, size = round(N * split, 0), replace = F)
  # Store indices in list
  split_indxs$train <- c(1:N)[train_indxs]
  split_indxs$test <- c(1:N)[-train_indxs]

  return(split_indxs)
}

# Function to perform train-test split with no stratification
.cv <- function(N, n_folds, random_seed) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)

  # Initialize list
  cv_indxs <- list()
  # Create folds; start with randomly shuffling indices
  indices <- sample(1:N)
  # Get floor
  folds_vec <- rep(floor(N / n_folds), n_folds)
  excess <- N - sum(folds_vec)
  if (excess > 0) {
    folds_vector <- rep(1:n_folds, excess)[1:excess]
    for (num in folds_vector) folds_vec[num] <- folds_vec[num] + 1
  }
  # Random shuffle
  folds_vec <- sample(folds_vec, size = length(folds_vec), replace = FALSE)
  for (i in 1:n_folds) {
    # Create fold with stratified or non stratified sampling
    fold_indxs <- indices[1:folds_vec[i]]
    # Remove rows from vectors to prevent overlapping, last fold may be smaller or larger than other folds
    indices <- indices[-c(1:folds_vec[i])]
    # Add indices to list
    cv_indxs[[sprintf("fold%s", i)]] <- fold_indxs
  }
  return(cv_indxs)
}

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
