"print.vswift"<- function(object){
  if(class(object) == "vswift"){
    #Print parameter information
    cat(sprintf("Model Type: %s\n\n", object[["information"]][["parameters"]][["model_type"]]))
    #Creating response variable
    cat(sprintf("Features: %s\n\n", paste(object[["information"]][["parameters"]][["features"]], collapse = ",")))
    cat(sprintf("Response variable: %s\n\n", object[["information"]][["parameters"]][["response_variable"]]))
    cat(sprintf("Classes: %s\n\n", paste(unlist(object[["classes"]]), collapse = ", ")))
    cat(sprintf("K: %s\n\n", object[["information"]][["parameters"]][["k"]]))
    cat(sprintf("Split: %s\n\n", object[["information"]][["parameters"]][["split"]]))
    cat(sprintf("Stratified Sampling: %s\n\n", object[["information"]][["parameters"]][["stratified"]]))
    cat(sprintf("Random Seed: %s\n\n", object[["information"]][["parameters"]][["random_seed"]]))
    #Print sample size and missing data for user transparency
    cat(sprintf("Missing Data: %s\n\n", object[["information"]][["parameters"]][["missing_data"]]))
    cat(sprintf("Sample Size: %s\n\n", object[["information"]][["parameters"]][["sample_size"]]))
  }
}