
library(mlr)
library(stringi)
library(BBmisc)
library(ggplot2)
library("OpenML")
library(mlrCPO)
library(iml)

load("data.rda")
print(summarizeColumns(data)[, -c(5, 6, 7)], digits = 0)

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
  c("Cabin","PassengerId", "Ticket", "Name"))

print(summarizeColumns(data)[, -c(5, 6, 7)], digits = 0)

data = impute(data, cols = list(
  Age = imputeMedian(),
  Fare = imputeMedian(),
  Embarked = imputeConstant("__miss__"),
  farePp = imputeMedian(),
  deck = imputeConstant("__miss__"),
  portside = imputeConstant("__miss__")
))

data = data$data
data = convertDataFrameCols(data, chars.as.factor = TRUE)

task = makeClassifTask(id = "titanic", data = data, 
  target = "Survived", positive = "1")

print(task)

lrn = makeLearner("classif.randomForest", 
  predict.type = "prob")

n = nrow(data)
train = sample(n, size = 2/3 * n)
test = setdiff(1:n, train)

mod = train(lrn, task, subset = train)

pred = predict(mod, task = task, subset = test)
head(as.data.frame(pred))

performance(pred, measures = list(mlr::acc, mlr::auc))

rdesc = makeResampleDesc("CV", iters = 3, 
  stratify = TRUE)

r = resample(lrn, task, rdesc, 
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

head(getBMRPerformances(bmr, as.df = TRUE), 4)

plotBMRBoxplots(bmr)

res = holdout(lrn, task)
df = generateThreshVsPerfData(res$pred, 
  list(fpr, tpr, acc))
plotROCCurves(df)

print(calculateROCMeasures(pred), abbreviations = FALSE)

lrn = makeLearner("classif.rpart")
getParamSet(lrn)

lrn = makeLearner("classif.ksvm", C = 5, sigma = 3)
lrn = setHyperPars(lrn, C = 1, sigma = 2)

lrn = makeLearner("classif.ksvm", 
  predict.type = "prob")

par.set = makeParamSet(
  makeNumericParam("C", lower = -8, upper = 8, 
    trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -8, upper = 8, 
    trafo = function(x) 2^x)
)

tune.ctrl = makeTuneControlRandom(maxit = 50L)
tr = tuneParams(lrn, task = task, par.set = par.set,
  resampling = rdesc, control = tune.ctrl,
  measures = mlr::auc)

head(as.data.frame(tr$opt.path))[, c(1,2,3,7)]

inner = makeResampleDesc("Subsample", iters = 4)
lrn = makeLearner("classif.ksvm", predict.type = "prob")
lrn.autosvm = makeTuneWrapper(
  lrn, resampling = inner, 
  par.set = par.set, control = tune.ctrl,
  measures = mlr::auc)

r = resample(lrn.autosvm, task, 
  resampling = rdesc, extract = getTuneResult,
  measures = mlr::auc)
r
r$extract

bmr2 = benchmark(lrn.autosvm, task, rdesc)

plotBMRBoxplots(mergeBenchmarkResults(list(bmr, bmr2)))

operation = cpoScale()
print(operation)

rm(list=ls(all=TRUE))
load("data.rda")
data$Name = as.factor(data$Name)
data = impute(data, cols = list(
  Fare = imputeMedian()
))
data = data$data

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
  impute.cpo %>>% makeLearner("classif.randomForest", 
  predict.type = "prob")

# the new object is a "CPOLearner", subclass of "Learner"
inherits(learner, "CPOLearner")

# train using the task that was not preprocessed
ppmod = train(learner, train.task)

lrn = cpoFilterFeatures(abs = 2L) %>>% 
  makeLearner("classif.randomForest")

ps = makeParamSet(
  makeDiscreteParam("filterFeatures.method", 
    values = c("anova.test", "chi.squared")),
  makeIntegerParam("mtry", lower = 1, upper = 10)
)
ctrl = makeTuneControlRandom(maxit = 10L)
tr = tuneParams(lrn, iris.task, cv3, par.set = ps, 
  control = ctrl)

scale = cpoSelect(pattern = "Fare", id = "first") %>>%
  cpoScale(id = "scale")
scale.pca = scale %>>% cpoPca()
cbinder = cpoCbind(scale, scale.pca, cpoSelect(
  pattern = "Age", id = "second"))
result = train.data %>>% cbinder
result[1:3, ]

mod

train.data = data[rownames(data) %in% train, ]

X = dropNamed(train.data, "Survived")
iml.mod = Predictor$new(mod, data = X, 
  y = train.data$Survived, class = 2)

imp = FeatureImp$new(iml.mod, loss = "ce")
plot(imp)

pdp = PartialDependence$new(iml.mod, feature = "Pclass")
plot(pdp)

X[1,]

lime = LocalModel$new(iml.mod, x.interest = X[1,])
plot(lime)

datasets = listOMLDataSets() 
datasets[1:3, c(1,2,11)]

tasks = listOMLTasks()
tasks[1:3, 1:4]

listOMLDataSets(data.name = "titanic")[, 1:5]
titanic = getOMLDataSet(data.id = 40945L)

listOMLTasks(data.name = "titanic")[1:2, 1:4]
titanic.task = getOMLTask(task.id = 146230)
titanic.task

lrn = makeLearner("classif.randomForest", mtry = 2)
run.mlr = runTaskMlr(titanic.task, lrn)
run.mlr$bmr$results




