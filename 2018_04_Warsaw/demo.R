# Link to all stuff: https://github.com/mlr-org/mlr/wiki/mlr-for-Warsaw-2018

library(mlr)
library(stringi)
library(BBmisc)
library(ggplot2)
library(parallelMap)
library(iml)

load("data.rda")
print(summarizeColumns(data)[,-c(5, 6, 7)], digits = 0)



# FEATURE ENGINEERING
data$Embarked[data$Embarked == ""] = NA
data$Embarked = droplevels(data$Embarked)
data$Cabin[data$Cabin == ""] = NA
data$Cabin = droplevels(data$Cabin)

# Price per person, multiple tickets bought by one
# person
data$farePp = data$Fare / (data$Parch + data$Sibsp + 1)

# The deck can be extracted from the the cabin number
data$deck = as.factor(stri_sub(data$Cabin, 1, 1))

# Starboard had an odd number, portside even cabin
# numbers
data$portside = stri_sub(data$Cabin, 3, 3)
data$portside = as.numeric(data$portside) %% 2

# Drop stuff we cannot easily model on
data = dropNamed(data,
  c("Cabin", "PassengerId", "Ticket", "Name"))

print(summarizeColumns(data)[, -c(5, 6, 7)], digits = 0)

data = impute(
  data,
  cols = list(
  Age = imputeMedian(),
  Fare = imputeMedian(),
  Embarked = imputeConstant("__miss__"),
  farePp = imputeMedian(),
  deck = imputeConstant("__miss__"),
  portside = imputeConstant("__miss__")
  )
)

data = data$data
data = convertDataFrameCols(data, chars.as.factor = TRUE)



# TASK - TRAIN - EVAL
task = makeClassifTask(
  id = "titanic",
  data = data,
  target = "Survived",
  positive = "1"
)
print(task)

lrn.rf = makeLearner("classif.randomForest", 
  predict.type = "prob")

n = nrow(data)
train = sample(n, size = 2 / 3 * n)
test = setdiff(1:n, train)

mod = train(lrn.rf, task, subset = train)

pred = predict(mod, task = task, subset = test)
head(as.data.frame(pred))

performance(pred, measures = list(mlr::acc, mlr::auc))

rdesc = makeResampleDesc("CV", iters = 3, stratify = TRUE)

r = resample(lrn.rf, task, rdesc,
  measures = list(mlr::acc, mlr::auc))
print(r)

head(r$measures.test)
head(as.data.frame(r$pred))

set.seed(3)
learners = c("glmnet", "naiveBayes", "randomForest",
  "ksvm")
learners = makeLearners(learners, type = "classif",
  predict.type = "prob")

bmr = benchmark(learners, task, rdesc,
  measures = mlr::auc)

getBMRAggrPerformances(bmr, as.df = TRUE)

plotBMRBoxplots(bmr)



# TUNING
lrn.ksvm = makeLearner("classif.ksvm", predict.type = "prob")

par.set = makeParamSet(
  makeNumericParam("C", lower = -8, upper = 8, 
  trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -8, upper = 8, 
  trafo = function(x) 2^x)
)

tune.ctrl = makeTuneControlRandom(maxit = 10L)

tr = tuneParams(lrn.ksvm, task = task, par.set = par.set,
  resampling = rdesc, control = tune.ctrl,
  measures = mlr::auc)

head(as.data.frame(tr$opt.path))[, c(1, 2, 3, 7)]


classif.ksvm.tuned = makeTuneWrapper(
  lrn.ksvm, resampling = rdesc, 
  par.set = par.set, control = tune.ctrl)

bmr2 = benchmark(classif.ksvm.tuned, task, rdesc)
plotBMRBoxplots(mergeBenchmarkResults(list(bmr, bmr2)))




# NESTED RESAMPLING
inner.rdesc = makeResampleDesc("Subsample", iters = 2)

classif.ksvm.inner = makeTuneWrapper(
  lrn.ksvm, resampling = inner.rdesc, 
  par.set = par.set, control = tune.ctrl,
  measures = mlr::auc)

r.nest = resample(classif.ksvm.inner, task, 
  resampling = rdesc, extract = getTuneResult,
  measures = mlr::auc)
r.nest
r.nest$extract



# PERFORMANCE
res = holdout(lrn.rf, task)
df = generateThreshVsPerfData(res$pred, list(fpr, tpr, acc))
plotROCCurves(df)

print(calculateROCMeasures(pred), abbreviations = FALSE)



# IML
train.data = data[rownames(data) %in% train,]

X = dropNamed(train.data, "Survived")
iml.mod = Predictor$new(mod, data = X,
  y = train.data$Survived, class = 2)

imp = FeatureImp$new(iml.mod, loss = "ce")
plot(imp)

pdp = PartialDependence$new(iml.mod, feature = "Pclass")
plot(pdp)

X[1, ]

lime = LocalModel$new(iml.mod, x.interest = X[1, ])
plot(lime)




# MBO
library(mlrMBO) # Bayesian Optimization in R
library(ParamHelpers) # Objects for parameter spaces
library(smoof) # Interface for objective functions
set.seed(2)

iters = 5

par.set = makeParamSet(
  makeNumericParam("C", lower = -8, upper = 8, 
  trafo = function(x) 2^x), 
  makeNumericParam("sigma", lower = -8, upper = 8, 
  trafo = function(x) 2^x)
)

svm = makeSingleObjectiveFunction(name = "svm.tuning", 
  fn = function(x) {
  # remove inactive parameters coded with `NA`
  x = x[!vlapply(x, is.na)]
  lrn = makeLearner("classif.ksvm", par.vals = x)
  crossval(lrn, task, iters = 2, show.info = FALSE)$aggr
  },
  par.set = par.set,
  noisy = TRUE,
  has.simple.signature = FALSE,
  minimize = TRUE
)

ctrl = makeMBOControl()
ctrl = setMBOControlTermination(ctrl, iters = iters)

makeMBOLearner(ctrl, svm)

res = mbo(svm, control = ctrl)
print(res)

op = as.data.frame(res$opt.path)
plot(cummin(op$y), type = "l", ylab = "mmce", 
  xlab = "iteration")




# CPO
library(mlrCPO)

rm(list=ls(all=TRUE))
load("data.rda")
data$Name = as.factor(data$Name)

n = nrow(data)
train = sample(n, size = 2/3 * n)
test = setdiff(1:n, train)

train.data = data[rownames(data) %in% train, ]
test.data = data[rownames(data) %in% test, ]

# Add interesting columns
newcol.cpo = cpoAddCols(
  farePp = Fare / (Parch + Sibsp + 1),
  deck = stri_sub(Cabin, 1, 1),
  side = {
  digit = stri_sub(Cabin, 3, 3)
  digit = suppressWarnings(as.numeric(digit))
  c("port", "starboard")[digit %% 2 + 1]
  })

# drop uninteresting columns
dropcol.cpo = cpoSelect(names = c("Cabin",
  "Ticket", "Name"), invert = TRUE)

# impute
impute.cpo = cpoImputeMedian(affect.type = "numeric") %>>%
  cpoImputeConstant("__miss__", affect.type = "factor")

train.task = makeClassifTask("Titanic", train.data,
  target = "Survived")

pp.task = train.task %>>% newcol.cpo %>>%
  dropcol.cpo %>>% impute.cpo

# get retransformation
ret = retrafo(pp.task)
# can be applied to data using the %>>% operator,
# just as a normal CPO
pp.test = test.data %>>% ret

learner = newcol.cpo %>>% dropcol.cpo %>>%
  impute.cpo %>>% makeLearner("classif.randomForest")

# the new object is a "CPOLearner", subclass of "Learner"
inherits(learner, "CPOLearner")

# train using the task that was not preprocessed
ppmod = train(learner, train.task)




# OPENML
library("OpenML")

setOMLConfig(apikey = "...")

# Permanently save your API disk to your config file
saveOMLConfig(apikey = "...", overwrite=TRUE)

listOMLDataSets(data.name = "titanic")[, 1:5]
titanic = getOMLDataSet(data.id = 40945L)
print(summarizeColumns(titanic$data)[, -c(5, 6, 7)], digits = 0)

listOMLTasks(data.name = "titanic")[1:2, 1:4]
titanic.task = getOMLTask(task.id = 146230)
titanic.task

lrn = makeLearner("classif.randomForest", mtry = 2)
run.mlr = runTaskMlr(titanic.task, lrn)
run.mlr$bmr$results

titanic.desc = makeOMLDataSetDescription(name = "titanic", 
  description = "Titanic data set ...")

titanic.data = makeOMLDataSet(desc = titanic.desc, 
  data = data, target.features = "Survived")

# titanic.id = uploadOMLDataSet(titanic.data)
