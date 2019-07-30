# List of learning tasks
tasks <- mlr_tasks$mget(c("iris", "spam"))

learner = mlr_learners$mget(c("classif.kknn", "classif.svm"))
measures = mlr_measures$mget("classif.ce")

## ParamSets

param_set_svm = {
  foo = ParamSet$new(params = list(
    ParamDbl$new("cost", lower = -5, upper = 5),
    ParamDbl$new("gamma", lower = -5, upper = 3)
  ))

  foo$trafo = function(x, param_set) {
    x$cost = 2^x$cost
    x$gamma = 2^x$gamma
    return(x)
  }
}

param_set_knn = ParamSet$new(params = list(
  ParamDbl$new("k", lower = 10, upper = 50),
  ParamDbl$new("distance", lower = 1, upper = 50)
))

terminator = TerminatorEvaluations$new(100)
resampling_inner = mlr_resamplings$get("cv3")

# AutoTuner SVM

## at_svm only contains "TRUE" instead of the R6 object :(
at_svm = {
  foo = AutoTuner$new(learner$classif.svm, resampling_inner, measures = measures,
                       param_set_svm, terminator, tuner = TunerRandomSearch)
foo$store_bmr = TRUE
}

# AutoTuner KNN

# at_knn$store_bmr = FALSE :(
at_knn = {
  foo = AutoTuner$new(learner$classif.kknn, resampling_inner, measures = measures,
                      param_set_knn, terminator, tuner = TunerRandomSearch)
  #foo$store_bmr = TRUE
}
at_knn$store_bmr = TRUE

# Resampling
resampling_outer = mlr_resamplings$get("repeated_cv", param_vals = list(folds = 5, repeats = 10))

# BM design
design = expand_grid(tasks, list(at_svm, at_knn), resampling_outer)

# execute benchmark
bm = {
  plan("multicore", workers = 5)
  benchmark(design = design)
}
