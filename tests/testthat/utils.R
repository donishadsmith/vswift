library(testthat)

check_png <- function() {
  for (png_file in list.files(getwd(), pattern = ".png")) {
    expect_true(file.size(png_file) > 0)
    file.remove(png_file)
  }

  file.remove(list.files(getwd(), pattern = "Rplots.pdf"))
}

# Ensure all metrics in interval [0,1]
check_conditions <- function(x, curve) {
  if (curve == "roc") {
    met_names <- c("tpr", "fpr", "youdins_indx")
  } else {
    met_names <- c("precision", "recall", "maxF1", "optimal_threshold")
  }

  met_names <- c(met_names, "auc", "thresholds", "probs")

  for (name in met_names) {
    if (name %in% c("tpr", "fpr", "precision", "recall")) {
      expect_true(all(x$metrics[[name]] >= 0))
      expect_true(all(x$metrics[[name]] <= 1))
    } else {
      expect_true(all(x[[name]] >= 0))
      expect_true(all(x[[name]] <= 1))
    }
  }
}


check_metrics <- function(out, curve) {
  for (mod in names(out)) {
    for (split_method in names(out[[mod]])) {
      for (id in names(out[[mod]][[split_method]])) {
        x <- out[[mod]][[split_method]][[id]]
        check_conditions(x, curve)
      }
    }
  }
}
