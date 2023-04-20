.stratified_check <- function(class,class_indices,output,n){
  if(round(n*output[["class_proportions"]][[class]],0) == 0){
    stop(sprintf("0 indices selected for %s class\n not enough samples for stratified sampling",class))
  }
  if(round(n*output[["class_proportions"]][[class]],0) > length(class_indices[[class]])){
    stop(sprintf("not enough samples of %s class for stratified sampling",class))
  }
}