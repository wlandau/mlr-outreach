### Not-so-basic hyperparameter tuning: nested resampling

library(methods)
library(mlbench)
library(mlr)

# yeah, same data again and knn
data = mlbench.threenorm(1000)
task = makeClassifTask(data = as.data.frame(data), target = "classes")
lrn = makeLearner("classif.kknn")

# our goal: tune k, but do a proper evaluation
par.set = makeParamSet(
  makeDiscreteParam("k", values = 1:10)
)

# cross-validation, for inner tuning
inner = makeResampleDesc("CV", iters = 5)
# subsampling for outer evaluation
outer = makeResampleDesc("Subsample", iters = 10, split = 0.8)


# very similar as tuneParams before
# but now we construct a wrapper, that does tuning internally,
# then selects the best param, then fits on the outer training data
ctrl = makeTuneControlGrid()
lrn2 = makeTuneWrapper(lrn, resampling = inner, par.set = par.set, control = ctrl)

# simply doing this now does full nested sampling
r = resample(lrn2, task, resampling = outer)

# access full info of result
# (well it is structurally the same object we know from ex2.R ...)
print(names(r))
print(r$measures.test)
print(r$aggr)
print(head(as.data.frame(r$pred)))

# hmm, great, but I want the result of the 10 tunings!
# where are they?

# answer: they are stored in the model of the trained TuneWrapper
# we can use the "extract" option of "resample" to access this (and much more)

r = resample(lrn2, task, resampling = outer, extract = function(model) getTuneResult(model))
print(length(r$extract))
print(r$extract[[1]])


