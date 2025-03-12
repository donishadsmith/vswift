# Helper function to calculate metrics
.calculate_metrics <- function(class, ground, pred) {
  # Sum of true positives
  true_pos <- sum(ground[ground == class] == pred[ground == class])
  # Sum of false negatives
  false_neg <- sum(ground == class & pred != class)
  # Sum of the false positive
  false_pos <- sum(pred == class) - true_pos
  # Calculate metrics
  precision <- true_pos / (true_pos + false_pos)
  recall <- true_pos / (true_pos + false_neg)
  f1 <- 2 * (precision * recall) / (precision + recall)

  metrics <- list("precision" = precision, "recall" = recall, "f1" = f1)
  return(metrics)
}

# Helper function to populate metrics dataframes
.populate_metrics_df <- function(id, classes, vec, met_df) {
  # Create variables used in for loops to calculate precision, recall, and f1

  # Variable to select correct dataframe
  col <- ifelse(id == "split", "Set", "Fold")

  dict <- list("train" = "Training", "test" = "Test")

  for (j in names(vec$pred)) {
    # Calculate classification accuracy
    class_acc <- sum(vec$ground[[j]] == vec$pred[[j]]) / length(vec$ground[[j]])
    rowid <- ifelse(col == "Set", dict[[j]], paste(c(col, unlist(strsplit(id, split = "fold"))[2]), collapse = " "))
    met_df[met_df[, col] == rowid, "Classification Accuracy"] <- class_acc

    # class specific metrics
    for (class in classes) {
      met_list <- .calculate_metrics(class, vec$ground[[j]], vec$pred[[j]])
      # Add information to dataframes
      met_df[met_df[, col] == rowid, sprintf("Class: %s Precision", class)] <- met_list$precision
      met_df[met_df[, col] == rowid, sprintf("Class: %s Recall", class)] <- met_list$recall
      met_df[met_df[, col] == rowid, sprintf("Class: %s F1", class)] <- met_list$f1

      # Warning is a metric is NA
      met_vals <- c(
        "classification accuracy" = class_acc, "precision" = met_list$precision,
        "recall" = met_list$recall, "f1" = met_list$f1
      )
      if (any(is.na(met_vals))) {
        warning(sprintf(
          "For class %s - %s, the following metrics could not be calculated: '%s'",
          class, rowid, paste(names(which(is.na(met_vals))), collapse = "', '")
        ))
      }
    }
  }
  if (col == "Fold") met_df <- met_df[met_df[, col] == rowid, colnames(met_df)]

  return(met_df)
}

# Helper function to merge cv dfs
.merge_df <- function(iters, metrics, cv_df) {
  for (i in seq_along(iters)) {
    cv_df[i, colnames(cv_df)] <- metrics[[iters[[i]]]][1, ]
  }

  return(cv_df)
}

# Calculate mean, standard deviation, and standard error for cross validation
.get_desc <- function(cv_df, n_folds) {
  indx <- nrow(cv_df)
  desc <- c("Mean CV:", "Standard Deviation CV:", "Standard Error CV:")
  cv_df[(indx + 1):(indx + 3), "Fold"] <- desc
  for (colname in colnames(cv_df)[colnames(cv_df) != "Fold"]) {
    # Create vector containing corresponding column name values for each fold
    num_vector <- cv_df[1:indx, colname]
    cv_df[cv_df$Fold == desc[1], colname] <- mean(num_vector, na.rm = TRUE)
    cv_df[cv_df$Fold == desc[2], colname] <- sd(num_vector, na.rm = TRUE)
    cv_df[cv_df$Fold == desc[3], colname] <- sd(num_vector, na.rm = TRUE) / sqrt(n_folds)
  }

  return(cv_df)
}
