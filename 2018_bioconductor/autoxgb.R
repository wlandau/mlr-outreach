library(mlr)
library(mlrCPO)
library(mlrMBO)
library(parallelMap)

# lets build autoxgboost in < 50 lines of code :)


# lets work on the titanic data set
# - has factors
# - has NAs
titanic = read.csv("titanic_train.csv")
# drop some annoying string features
titanic = dropNamed(titanic, c("Ticket", "Cabin"))
task = makeClassifTask("titanic", titanic, target = "Survived")

# create xgboost as a powerful learner
lrn = makeLearner("classif.xgboost")

# build a pipeline that handles:
# - NA imputation (mean for nums, mode for factors)
# - different types of factor encoding
# - different types of feature filtering
pipeline = cpoImputeAll(classes = list(numeric = imputeMean(), factor = imputeMode()))
pipeline = pipeline %>>% cpoMultiplex(id = "factenc", 
  cpos = list(cpoDummyEncode(), cpoImpactEncodeClassif()))
pipeline = pipeline %>>% cpoFilterFeatures()
pipeline = pipeline %>>% lrn


# define the param space, we want to tune over.

ps = makeParamSet(
  makeDiscreteParam("factenc.selected.cpo", 
    values = c("dummyencode", "impact.encode.classif")),
  makeDiscreteParam("filterFeatures.method", 
    values = c("anova.test", "auc")),
  makeNumericParam("filterFeatures.perc", lower = 0.1, upper = 1),
  makeNumericParam("alpha", lower = -10, upper = 10, 
    trafo = function(x) 2^x),
  makeIntegerParam("nrounds", lower = 1, upper = 100)
)

# we want Bayesian optimization for effiecient configuration
ctrl = makeTuneControlMBO(budget = 2)

# attach autotuning to pipeline-xgboost 
autoxgb = makeTuneWrapper(pipeline, cv3, par.set = ps, control = ctrl)

# Nested crossvalidation in parallel
# parallelStartSocket()
r = resample(autoxgb, task, cv3)
# parallelStop()
