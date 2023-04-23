.check_additional_arguments <<- function(model_type = NULL,...){
    additional_args <- names(list(...))
    switch(model_type,
           "lda" = {
             valid_args <- c("grouping","prior","method","nu")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "qda" = {
             valid_args <- c("grouping","prior","method","nu")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "logistic" = {
             valid_args <- c("weights","start","etastart","mustart","offset","control","contrasts","intercept","singular.ok","typw")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "svm" = {
             valid_args <- c("scale","type","kernal","degree","gamma","coef0","cost","nu","class.weights","cachesize","tolerance","epsilon",
                             "shrinking","cross","probability","fitted")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "naivebayes" = {
             valid_args <- c("prior","laplace","usekernel","usepoisson")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "nnet" = {
             valid_args <- c("weights","size","Wts","mask","linout","entropy","softmax","censored","skip","rang","decay","maxit",
                             "Hess","trace","MaxNWts","abstol","reltol")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           "knn" = {
             valid_args <- c("kmax","ks","kmax","distance","kernel","scale","contrasts","ykernel")
             invalid_args <- additional_args[which(!additional_args %in% valid_args)]},
           
    )
  if(length(invalid_args) > 0){
    if(model_type %in% c("lda","qda")){
      stop(sprintf("the following arguments are invalid for %s or is incompatable with categorical_cv_split: %s",paste0(model_type,".default"),paste(invalid_args,collapse = ",")))
    }else if(model_type %in% c("logistic","linear")){
      stop(sprintf("the following arguments are invalid for glm function or is incompatable with categorical_cv_split: %s",model_type,paste(invalid_args,collapse = ",")))
    }else if(model_type == "svm"){
      stop(sprintf("the following arguments are invalid for  %s or is incompatable with categorical_cv_split: %s",paste0(model_type,".default"),paste(invalid_args,collapse = ",")))
    }else if(model_type == "naivebayes"){
      stop(sprintf("the following arguments are invalid for naive_bayes.default or is incompatable with %s: %s",paste(invalid_args,collapse = ",")))
    }else if(model_type == "nnet"){
      stop(sprintf("the following arguments are invalid for %s or is incompatable with categorical_cv_split: %s",paste0(model_type,".default"),paste(invalid_args,collapse = ",")))
    }else if(model_type == "knn"){
      stop(sprintf("the following arguments are invalid for train.kknn or is incompatable with categorical_cv_split: %s",paste(invalid_args,collapse = ",")))
    }
  }
}