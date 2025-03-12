# Helper function to obtain class specific information
.get_class_info <- function(target_vector) {
  # Get proportions and indices
  prop_dict <- table(target_vector) / sum(table(target_vector))
  indices_dict <- list()
  # Get the indices with the corresponding categories and add to list
  for (class in names(prop_dict)) {
    indices_dict[[class]] <- which(target_vector == class)
  }

  return(list("proportions" = prop_dict, "indices" = indices_dict))
}

# Helper function to get class proportions in data partitions
.get_proportions <- function(target_vector, indxs) {
  prop_dict <- list()

  for (id in names(indxs)) {
    vec_subset <- table(target_vector[indxs[[id]]])
    prop_dict[[id]] <- vec_subset / sum(vec_subset)
  }

  return(prop_dict)
}

# Function for stratified train-test split
.stratified_split <- function(classes, class_indxs, class_props, N, split, random_seed) {
  # Set seed
  if (!is.null(random_seed)) set.seed(random_seed)
  # Split sizes
  train_n <- round(N * split, 0)
  test_n <- N - train_n
  # Initialize list
  split_indxs <- list("train" = NULL, "test" = NULL)

  for (class in classes) {
    # Check if sampling possible
    .stratified_check(class, class_indxs[[class]], class_props[[class]], train_n, "train-test splitting", "training set")
    # Get train indices
    train_indxs <- sample(class_indxs[[class]], size = floor(train_n * class_props[[class]]), replace = F)
    # Store indices for train set
    split_indxs$train <- c(split_indxs$train, train_indxs)
    # Remove indices to not add to test set
    class_indxs[[class]] <- class_indxs[[class]][!(class_indxs[[class]] %in% train_indxs)]
    # Check if the test set will have the necessary proportions
    .stratified_check(class, class_indxs[[class]], class_props[[class]], test_n, "train-test splitting", "test set")
    # Add indices for test set
    test_indxs <- sample(class_indxs[[class]], size = floor(test_n * class_props[[class]]), replace = F)
    split_indxs$test <- c(split_indxs$test, test_indxs)
    # Remove indices to check if leftovers remain for special circumstances
    class_indxs[[class]] <- class_indxs[[class]][!(class_indxs[[class]] %in% test_indxs)]
  }

  # Get number of leftover
  leftover <- length(as.vector(unlist(class_indxs)))
  if (leftover > 0) split_indxs <- .excess_split(class_indxs, split_indxs, classes)

  return(split_indxs)
}

.excess_split <- function(class_indxs, split_indxs, classes) {
  # Check relative proportion; train will be higher than test
  test_prop <- length(split_indxs$test) / length(split_indxs$train)
  train_prop <- 1 - test_prop

  for (class in classes) {
    if (length(class_indxs[class]) > 0) {
      train_indxs <- sample(class_indxs[[class]], size = ceiling(length(class_indxs[[class]] * train_prop)))
      split_indxs$train <- c(split_indxs$train, train_indxs)
      class_indxs[[class]] <- class_indxs[[class]][!(class_indxs[[class]] %in% train_indxs)]
      # Assign rest to test
      split_indxs$test <- c(split_indxs$test, class_indxs[[class]])
    }
  }

  return(split_indxs)
}

# Function for stratified cv
.stratified_cv <- function(classes, class_indxs, class_props, N, n_folds, random_seed, strategy = "cross validation") {
  if (!is.null(random_seed)) set.seed(random_seed)
  # Initialize list
  cv_indxs <- list()

  # Create folds
  for (i in 1:n_folds) {
    # Keep initializing variable
    fold_indxs <- c()
    # Fold size; try to undershoot for excess
    fold_n <- floor(N / n_folds)

    # Assign class indices to each fold
    for (class in classes) {
      # Check if sampling possible
      .stratified_check(class, class_indxs[[class]], class_props[[class]], fold_n, strategy)
      sampled_indxs <- sample(class_indxs[[class]], size = floor(fold_n * class_props[[class]]), replace = F)
      fold_indxs <- c(fold_indxs, sampled_indxs)
      # Remove already selected indices
      class_indxs[[class]] <- class_indxs[[class]][-which(class_indxs[[class]] %in% sampled_indxs)]
    }
    # Add indices to list
    cv_indxs[[sprintf("fold%s", i)]] <- fold_indxs
  }
  # Deal with excess
  if (N - length(as.numeric(unlist(cv_indxs))) > 0) cv_indxs <- .excess_cv(cv_indxs, class_indxs, classes, n_folds)

  return(cv_indxs)
}

# Function to deal with excess indices
.excess_cv <- function(cv_indxs, leftover, classes, n_folds) {
  for (class in classes) {
    # Get the remaining indices for class
    remain_idxs <- leftover[[class]]
    if (length(remain_idxs) > 0) {
      # Create a sequence of repeating fold ids c(1:5, 1:5, 1:5, ...) if n_fold is 5
      fold_seq <- rep(1:n_folds, length(remain_idxs))
      # Then truncate it to the length of the remaining class indices
      fold_seq <- fold_seq[1:length(remain_idxs)]
      # Assign one excess to the each id in the sequence
      for (i in seq_along(fold_seq)) {
        # Add indices to list
        fold_id <- sprintf("fold%s", fold_seq[i])
        cv_indxs[[fold_id]] <- c(remain_idxs[i], cv_indxs[[fold_id]])
      }
    }
  }

  return(cv_indxs)
}

# Helper function for .stratified_sampling to error check
.stratified_check <- function(class, class_indx, class_prop, N, strategy, id = NULL) {
  # Check if there are enough indices in class for proper assignment
  need <- floor(N * class_prop)
  have <- length(class_indx)

  if (strategy == "train-test splitting") {
    msg <- sprintf("need %s for %s but only have %s", need, id, have)
  } else {
    msg <- sprintf("need %s for each fold but only have %s", need, have)
  }

  if (need > length(class_indx)) {
    stop(sprintf(
      "not enough samples of %s class for %s using stratified sampling; %s",
      class, strategy, msg
    ))
  }
}
