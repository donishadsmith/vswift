.check_additional_arguments <<- function(model_type = NULL, call = NULL,...){
  if(call == "categorical_cv_split"){
    additional_args <- names(list(...))
    switch(model_type,
           "lda" = {
             valid_args <- names(formals(MASS:::lda.default))[!names(formals(MASS:::lda.default)) %in% c("CV","x")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "qda" = {
             valid_args <- names(formals(MASS:::qda.default))[!names(formals(MASS:::qda.default)) %in% c("CV","x")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "logistic" = {
             valid_args <- names(formals(glm))[!names(formals(glm)) %in% c("formula","data","family")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "svm" = {
             valid_args <- names(formals(e1071:::svm.default))[!names(formals(e1071:::svm.default)) %in% c("x","y")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "naivebayes" = {
             valid_args <- names(formals(naivebayes:::naive_bayes.default))[!names(formals(naivebayes:::naive_bayes.default)) %in% c("x","y")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "nnet" = {
             valid_args <- names(formals(nnet:::nnet.default))[!names(formals(nnet:::nnet.default)) %in% c("x","y")]
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]}
    )
  }
  if(length(invalid_args) > 0){
    if(model_type %in% c("lda","qda")){
      stop(sprintf("the following arguments are invalid for %s or is incompatable with %s: %s",paste0(model_type,".default"),call,paste(invalid_args,collapse = ",")))
    }else if(model_type %in% c("logistic","linear")){
      stop(sprintf("the following arguments are invalid for glm function or is incompatable with %s: %s",model_type,call,paste(invalid_args,collapse = ",")))
    }else if(model_type == "svm"){
      stop(sprintf("the following arguments are invalid for  %s or is incompatable with %s: %s",paste0(model_type,".default"),call,paste(invalid_args,collapse = ",")))
    }else if(model_type == "naivebayes"){
      stop(sprintf("the following arguments are invalid for naive_bayes.default or is incompatable with %s: %s",call,paste(invalid_args,collapse = ",")))
    }else if(model_type == "nnet"){
      stop(sprintf("the following arguments are invalid for %s or is incompatable with %s: %s",paste0(model_type,".default"),call,paste(invalid_args,collapse = ",")))
    }
  }
}