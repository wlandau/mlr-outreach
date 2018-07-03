# ---------------------------------------------------------------------------- #
# mlr_quickstart                                                               #
# ---------------------------------------------------------------------------- #

library(mlr)
task = sonar.task
n = getTaskSize(task)
lrn = makeLearner("classif.rpart", predict.type = "prob")
mod = train(lrn, task, subset = seq(1, n, 2))
pred = predict(mod, task = task, subset = seq(2, n, 2))
performance(pred, measures = list(mmce, mlr::auc))

rdesc = makeResampleDesc("CV", iters = 3L, stratify = TRUE)
r = resample(lrn, task, rdesc)
print(r$aggr)
print(r$measures.test)
print(head(as.data.frame(r$pred), 3L))

lrns = list(
  makeLearner("classif.rpart"),
  makeLearner("classif.randomForest")
)
b = benchmark(lrns, task, cv2, measures = mmce)
print(b)

print(getBMRAggrPerformances(b, as.df = TRUE))

print(getBMRPerformances(b, as.df = TRUE))
print(head(getBMRPredictions(b, as.df = TRUE), 3L))


# ---------------------------------------------------------------------------- #
# tuning_mlr                                                                   #
# ---------------------------------------------------------------------------- #

## ------------------------------------------------------------------------
lrn = makeLearner("classif.rpart")
getParamSet(lrn)

## ------------------------------------------------------------------------
lrn = makeLearner("classif.ksvm", C = 5, sigma = 3)
lrn = setHyperPars(lrn, C = 1, sigma = 2)

## ------------------------------------------------------------------------
lrn = makeLearner("classif.ksvm",
  predict.type = "prob")

# this is actually a bad way to encode the SVM space, see a few slides later
# how to do this properly
par.set = makeParamSet(
  makeNumericParam("C", lower = 0.001, upper = 100),
  makeNumericParam("sigma", lower = 0.001, upper = 100)
)

tune.ctrl = makeTuneControlRandom(maxit = 50L)
tr = tuneParams(lrn, task = task, par.set = par.set,
  resampling = hout, control = tune.ctrl,
  measures = mlr::auc)

## ------------------------------------------------------------------------
tr$x
tr$y
head(as.data.frame(tr$opt.path), 3L)[, c(1,2,3,7)]

lrn = makeLearner("classif.ksvm")
ps = makeParamSet(
  makeDiscreteParam("kernel", values = c("polydot", "rbfdot")),
  makeNumericParam("C", lower = -15, upper = 15,
    trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -15, upper = 15,
    trafo = function(x) 2^x,
   requires = quote(kernel == "rbfdot")),
  makeIntegerParam("degree", lower = 1, upper = 5,
   requires = quote(kernel == "polydot"))
)


# ---------------------------------------------------------------------------- #
# nested_resampling_04_demo                                                    #
# ---------------------------------------------------------------------------- #

lrn = makeLearner("classif.xgboost")
ps = makeParamSet(
  makeIntegerParam("nrounds", lower = 50, upper = 300),
  makeNumericParam("eta", lower = -5, upper = -0.01,
    trafo = function(x) 2^x)
)

ctrl = makeTuneControlRandom(maxit = 20)

# this adds the tuning to the learner,
# we use holdout on inner resampling
inner = makeResampleDesc(method = "Holdout")
lrn2 = makeTuneWrapper(lrn, inner, par.set = ps,
  control = ctrl, measures = mmce)

# now run everything, we use CV with 2 folds
# on the outer loop
outer = makeResampleDesc(method = "CV", iters = 2)
r = resample(lrn2, sonar.task, outer,
  extract = getTuneResult)

# lets look at some results from the outer iterations
r$extract[[1]]$x
r$extract[[1]]$y
r$extract[[1]]$opt.path

# ---------------------------------------------------------------------------- #
# parallel                                                                     #
# ---------------------------------------------------------------------------- #

## ----include=FALSE, cache=FALSE------------------------------------------
library(mlr)
library(parallelMap)

lrns = list(makeLearner("classif.rpart"), makeLearner("classif.svm"))
rdesc = makeResampleDesc("Bootstrap", iters = 100)

parallelStartSocket(4)
bm = benchmark(learners = lrns, tasks = iris.task, resamplings = rdesc)
parallelStop()

parallelStartSocket(4, level = "mlr.resample")
bm = benchmark(learners = lrns, tasks = iris.task, resamplings = rdesc)
parallelStop()


## ------------------------------------------------------------------------
set.seed(1)
library(ggplot2); library(RColorBrewer)
lrn = makeLearner("classif.randomForest", ntree = 200)
lrn = makeRemoveConstantFeaturesWrapper(learner = lrn)
lrn = makeDownsampleWrapper(learner = lrn)
lrn = makeFilterWrapper(lrn, fw.method = "gain.ratio")
filterParams(getParamSet(lrn), tunable = TRUE, type = c("numeric", "integer"))

## ------------------------------------------------------------------------
ps = makeParamSet(
  makeNumericParam("fw.perc", lower = 0.1, upper = 1),
  makeNumericParam("dw.perc", lower = 0.1, upper = 1))
res = tuneParams(lrn, sonar.task, resampling = cv10, par.set = ps,
  control = makeTuneControlRandom(maxit = 5), show.info = TRUE)
res

# ---------------------------------------------------------------------------- #
# mlrcpo                                                                       #
# ---------------------------------------------------------------------------- #

library(mlrCPO)

## ------------------------------------------------------------------------
task = iris.task
task = task %>>% cpoScale(scale = FALSE) %>>% cpoPca() %>>%  # pca
  cpoFilterChiSquared(abs = 3) %>>%  # filter
  cpoModelMatrix(~ 0 + .^2)  # interactions
head(getTaskData(task))

pipeline = cpoImputeMax() %>>% cpoDummyEncode() %>>% cpoFilterVariance()
getParamSet(pipeline)

str(getHyperPars(pipeline))

pipeline = setHyperPars(pipeline, variance.perc = 0.5)
tsk = iris.task %>>% pipeline
tsk

## ------------------------------------------------------------------------
lrn1 = makeLearner("classif.logreg")
getLearnerProperties(lrn1)
lrn2 = pipeline %>>% lrn1
getLearnerProperties(lrn2)

