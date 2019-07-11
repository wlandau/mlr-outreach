### Even more complex example, that shows wrappers and compares them with benchmark
### show cases quite a lot of features together

library(mlr)
library(mlbench)
configureMlr(show.info = TRUE)


# 1) a simple LDA
lrn1 = makeLearner("classif.lda")

# 2) tuned svm as in example before
lrn2 = makeLearner("classif.svm")
ps = makeParamSet(
  makeDiscreteParam("kernel", values = c("polynomial", "radial")),
  makeNumericParam("cost", lower = -15, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -15, upper = 15, trafo = function(x) 2^x,
   requires = quote(kernel == "radial")),
  makeIntegerParam("degree", lower = 1, upper = 5,
   requires = quote(kernel == "polynomial"))
)
ctrl = makeTuneControlRandom(maxit = 20)
inner = makeResampleDesc(method = "Holdout")
lrn2 = makeTuneWrapper(lrn2, inner, par.set = ps, control = ctrl)

# 3) a linear SVM with feature filtering and tuning
lrn3 = makeLearner("classif.svm", kernel = "linear")
lrn3 = makeFilterWrapper(lrn3)
# see how param sets get joined
print(getParamSet(lrn3))

ps = makeParamSet(
  makeNumericParam("fw.perc", lower = 0.7, upper = 1),
  makeNumericParam("cost", lower = -15, upper = 15, trafo = function(x) 2^x)
)
ctrl = makeTuneControlRandom(maxit = 20)
inner = makeResampleDesc(method = "Holdout")
lrn3 = makeTuneWrapper(lrn3, inner, par.set = ps, control = ctrl)

###################
# run all on Ionosphere
learners = list(lrn1, lrn2, lrn3)
data(Ionosphere)
task = makeClassifTask(data = Ionosphere, target = "Class")
task = removeConstantFeatures(task)
rdesc = makeResampleDesc("CV", iters = 2L)
br = benchmark(learners, task, resampling = rdesc)
print(br)





