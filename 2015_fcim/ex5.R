### Basic hyperparameter comparison / tuning

library(methods)
library(mlbench)
library(mlr)

# lets use the data from the last lecture tutorial
# and repeat some of our exercises

data = mlbench.threenorm(1000)
task = makeClassifTask(data = as.data.frame(data), target = "classes")

# we will compare values of k know for the data set
lrn = makeLearner("classif.kknn")

# cross-validation, no need to pregenerate now.
# as this is sensible, tuneParams will do this for us automatically
rdesc = makeResampleDesc("CV", iters = 10)

# Description of our parameter space we want to grid-search ove
par.set = makeParamSet(
  makeDiscreteParam("k", values = 1:10)
)

# run it
# (actually, mlr supports many other tuner, so we need to select the tuner via a control object here)
ctrl = makeTuneControlGrid()
res = tuneParams(lrn, task, rdesc, par.set = par.set, control = ctrl)

# access full info of result
print(res)
print(names(res))
print(head(as.data.frame(res$opt.path)))


