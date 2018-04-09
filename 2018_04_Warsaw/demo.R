## ----opts,include=FALSE,cache=FALSE--------------------------------------
library(knitr)
library(BBmisc)
library(mlr)
library(ggplot2)
library(parallelMap)
library(tikzDevice)
library(data.table)
library(gridExtra)
library(survMisc)
library(mlrMBO)
library(mlrCPO)
library(iml)
library(randomForest)

options(width = 80)
configureMlr(show.info = FALSE)
configureMlr(show.learner.output = FALSE)
OPENML_EVAL = TRUE

knit_hooks$set(document = function(x) {
  # silence xcolor
  x = sub('\\usepackage[]{color}', '\\usepackage{xcolor}', x, fixed = TRUE)
  # add an noindent after hooks -> remove blank line
  x = gsub('(\\\\end\\{knitrout\\}[\n]+)', '\\1\\\\noindent ', x)
  x
})

opts_chunk$set(
   fig.path = "knitr/figures/",
   cache.path = "knitr/cache/",
   cache = TRUE,
   tidy = FALSE,
#   dev = 'tikz',
   external = TRUE,
   fig.align = "center",
   size = "normalsize",
   stop = TRUE,
   fig.width = 9 * 0.8,
   fig.height = 6 * 0.8,
   small.mar = TRUE,
   prompt = TRUE
)

## ----intro, child="intro.Rnw"--------------------------------------------

## ----model-standard,eval=FALSE-------------------------------------------
## model = fit(target ~ ., data = train.data, ...)
## predictions = predict(model, newdata = test.data, ...)

## ----gatherSummary,include=FALSE-----------------------------------------
ee = as.environment("package:mlr")
nl = table(sub("^makeRLearner\\.([[:alpha:]]+)\\..+", "\\1", methods("makeRLearner")))
nm = sapply(list(classif = listMeasures("classif"), regr = listMeasures("regr"), surv = listMeasures("surv"), cluster = listMeasures("cluster")), length) - 4

## ----libraries, echo=FALSE, eval=TRUE, warning=FALSE---------------------
library(mlr)
library(stringi)
library(BBmisc)
library(ggplot2)

## ----dataImport, eval=TRUE-----------------------------------------------
load("data.rda")
print(summarizeColumns(data)[, -c(5, 6, 7)], digits = 0)

## ----eval=TRUE-----------------------------------------------------------
data$Embarked[data$Embarked == ""] = NA
data$Embarked = droplevels(data$Embarked)
data$Cabin[data$Cabin == ""] = NA
data$Cabin = droplevels(data$Cabin)

## ----feat1, eval=TRUE, warning=FALSE-------------------------------------
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

## ------------------------------------------------------------------------
print(summarizeColumns(data)[, -c(5, 6, 7)], digits = 0)

## ------------------------------------------------------------------------
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

## ------------------------------------------------------------------------
task = makeClassifTask(id = "titanic", data = data, 
  target = "Survived", positive = "1")

## ------------------------------------------------------------------------
print(task)


## ----benchmark, child="benchmark.Rnw"------------------------------------

## ----listlrns1, eval=TRUE, warning=FALSE---------------------------------
listLearners("classif", properties = c("prob",
  "multiclass"))[1:5, c(1,4,13,16)]

## ----lrn-----------------------------------------------------------------
lrn = makeLearner("classif.randomForest", 
  predict.type = "prob")

## ----train---------------------------------------------------------------
n = nrow(data)
train = sample(n, size = 2/3 * n)
test = setdiff(1:n, train)

mod = train(lrn, task, subset = train)

## ----pred----------------------------------------------------------------
pred = predict(mod, task = task, subset = test)
head(as.data.frame(pred))

## ------------------------------------------------------------------------
performance(pred, measures = list(mlr::acc, mlr::auc))

## ----kfoldCV, eval=TRUE, echo=FALSE, fig.height=2------------------------
# par(mar = c(0, 0, 0, 0))
# plot(1, type = "n", xlim = c(0, 10), ylim = c(0, 1), axes = FALSE)
# rect(seq(0, 8, by = 2), 0, seq(2, 10, by = 2), 1)
# text(seq(1, 9, by = 2), 0.5, col = c(rep("red", 2), "green", rep("red", 2)),
#      c(rep("Train", 2), "Test", rep("Train", 2)))
par(mar = c(0, 0, 0, 0))
plot(1, type = "n", xlim = c(-2, 10), ylim = c(0, 3), axes = FALSE) #, main = "Example of 3-fold Cross Validation")
rect(seq(0, 4, by = 2), 0.1, seq(2, 6, by = 2), 1, col = c("#56B4E944","#56B4E944","#E69F0044"))
rect(seq(0, 4, by = 2), 1.1, seq(2, 6, by = 2), 2, col = c("#56B4E944","#E69F0044","#56B4E944"))
rect(seq(0, 4, by = 2), 2.1, seq(2, 6, by = 2), 3, col = c("#E69F0044", "#56B4E944","#56B4E944"))

text(seq(1, 5, by = 2), 2.55, col = c("#E69F00", "#56B4E9","#56B4E9"),
  rev(c("Train", "Train", "Test")))
text(seq(1, 5, by = 2), 1.55, col = c("#56B4E9","#E69F00","#56B4E9"),
  c("Train", "Test", "Train"))
text(seq(1, 5, by = 2), 0.55, col = c("#56B4E9","#56B4E9","#E69F00"),
  c("Train", "Train", "Test"))
text(rep(-1, 3), c(0,1,2) + 0.55, paste("Iteration", 3:1))
# for (i in 1:3) #text(8, 2 - (i - 1) + 0.55, bquote(paste("=> ",widehat(Err)(widehat(f)[D[train]^.(i)], D[test]^.(i)))), cex = 1.3)
 # text(8, 2 - (i - 1) + 0.55, bquote(paste("=> ",widehat(GE)[D[test]^.(i)])), cex = 1.3)

## ----resampl1e-----------------------------------------------------------
rdesc = makeResampleDesc("CV", iters = 3, 
  stratify = TRUE)

r = resample(lrn, task, rdesc, 
  measures = list(mlr::acc, mlr::auc))
print(r)

## ----resample2-----------------------------------------------------------
head(r$measures.test)
head(as.data.frame(r$pred))

## ----benchmarking, eval=FALSE--------------------------------------------
## bmr = benchmark(list.of.learners, list.of.tasks, rdesc)

## ----bmrTitanic, eval=TRUE-----------------------------------------------
set.seed(3)

learners = c("glmnet", "naiveBayes", "randomForest", 
  "ksvm")
learners = makeLearners(learners, type = "classif", 
  predict.type = "prob")

bmr = benchmark(learners, task, rdesc, 
  measures = mlr::auc)

## ------------------------------------------------------------------------
getBMRAggrPerformances(bmr, as.df = TRUE)

## ------------------------------------------------------------------------
head(getBMRPerformances(bmr, as.df = TRUE), 4)

## ----plotBMR, eval=TRUE, fig.height=4------------------------------------
plotBMRBoxplots(bmr)


## ----performance, child="performance.Rnw"--------------------------------

## ------------------------------------------------------------------------
res = holdout(lrn, task)
df = generateThreshVsPerfData(res$pred, 
  list(fpr, tpr, acc))
plotROCCurves(df)

## ------------------------------------------------------------------------
print(calculateROCMeasures(pred), abbreviations = FALSE)


## ----tuning, child="tuning.Rnw"------------------------------------------

## ------------------------------------------------------------------------
lrn = makeLearner("classif.rpart")
getParamSet(lrn)

## ------------------------------------------------------------------------
lrn = makeLearner("classif.ksvm", C = 5, sigma = 3)
lrn = setHyperPars(lrn, C = 1, sigma = 2)

## ----gridSearch, eval=TRUE, echo=FALSE, message=FALSE, results="hide", fig.height=5----
lrn = makeLearner("classif.ksvm", predict.type = "prob")
par.set = makeParamSet(
  makeNumericParam("C", lower = -15, upper = 15,
  trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -15, upper = 15, 
  trafo = function(x) 2^x)
)

ctrl.grid = makeTuneControlGrid(resolution = 7)
set.seed(1)
res.grid = tuneParams(lrn, task = task, par.set = par.set,
  resampling = rdesc, control = ctrl.grid,
  measures = mlr::auc)
opt.grid = as.data.frame(res.grid$opt.path)

gridSearch = ggplot(opt.grid, aes(x = sigma, y = C, size = 1-auc.test.mean))
gridSearch + geom_point(shape = 21 , col = "black", fill = "#56B4E9" , alpha = .6)

## ----randomSearch, eval=TRUE, echo=FALSE, message=FALSE, results="hide", fig.height=5----
tune.ctrl.pic = makeTuneControlRandom(maxit = 40L)
set.seed(1)
res.rs = tuneParams(lrn, task = task, par.set = par.set,
  resampling = rdesc, control = tune.ctrl.pic,
  measures = mlr::auc)
opt.grid = as.data.frame(res.rs$opt.path)
rndSearch = ggplot(opt.grid, aes(x = sigma, y = C, size = 1-auc.test.mean))
rndSearch + geom_point(shape = 21 , col = "black", fill = "#56B4E9" , alpha = .6) + scale_x_continuous()

## ------------------------------------------------------------------------
lrn = makeLearner("classif.ksvm", 
  predict.type = "prob")

par.set = makeParamSet(
  makeNumericParam("C", lower = -8, upper = 8, 
    trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -8, upper = 8, 
    trafo = function(x) 2^x)
)

## ------------------------------------------------------------------------
tune.ctrl = makeTuneControlRandom(maxit = 50L)
tr = tuneParams(lrn, task = task, par.set = par.set,
  resampling = rdesc, control = tune.ctrl,
  measures = mlr::auc)

## ------------------------------------------------------------------------
head(as.data.frame(tr$opt.path))[, c(1,2,3,7)]


## ----nested, child="nestedresample.Rnw"----------------------------------

## ------------------------------------------------------------------------

inner = makeResampleDesc("Subsample", iters = 4)
lrn = makeLearner("classif.ksvm", predict.type = "prob")
lrn.autosvm = makeTuneWrapper(
  lrn, resampling = inner, 
  par.set = par.set, control = tune.ctrl,
  measures = mlr::auc)

## ------------------------------------------------------------------------
r = resample(lrn.autosvm, task, 
  resampling = rdesc, extract = getTuneResult,
  measures = mlr::auc)
r

## ------------------------------------------------------------------------
r$extract

## ------------------------------------------------------------------------
bmr2 = benchmark(lrn.autosvm, task, rdesc)

## ----tuningBmrTitanic, eval=TRUE, fig.height=4---------------------------
plotBMRBoxplots(mergeBenchmarkResults(list(bmr, bmr2)))


## ----parallel, child="parallel.Rnw"--------------------------------------

## ----parallelMap,eval=FALSE----------------------------------------------
## parallelStart("multicore")
## benchmark(...)
## parallelStop()


## ----mbo, child="mbo.Rnw"------------------------------------------------



## ----cpo, child="cpo.Rnw"------------------------------------------------

## ------------------------------------------------------------------------
library(mlrCPO)

## ------------------------------------------------------------------------
operation = cpoScale()
print(operation)

## ----eval=FALSE----------------------------------------------------------
## imputing.pca = cpoImputeMedian() %>>% cpoPca()

## ----eval=FALSE----------------------------------------------------------
## task %>>% imputing.pca

## ----eval=FALSE----------------------------------------------------------
## pca.rf = imputing.pca %>>%
##   makeLearner("classif.randomForest")

## ----include=FALSE-------------------------------------------------------
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

## ------------------------------------------------------------------------
# Add interesting columns
newcol.cpo = cpoAddCols(
  farePp = Fare / (Parch + Sibsp + 1),
  deck = stri_sub(Cabin, 1, 1),
  side = {
  digit = stri_sub(Cabin, 3, 3)
  digit = suppressWarnings(as.numeric(digit))
  c("port", "starboard")[digit %% 2 + 1]
  })

## ------------------------------------------------------------------------
# drop uninteresting columns
dropcol.cpo = cpoSelect(names = c("Cabin",
  "Ticket", "Name"), invert = TRUE)

# impute
impute.cpo = cpoImputeMedian(affect.type = "numeric") %>>%
  cpoImputeConstant("__miss__", affect.type = "factor")

## ----warning=FALSE-------------------------------------------------------
train.task = makeClassifTask("Titanic", train.data,
  target = "Survived")

pp.task = train.task %>>% newcol.cpo %>>%
  dropcol.cpo %>>% impute.cpo

## ------------------------------------------------------------------------
# get retransformation
ret = retrafo(pp.task)
# can be applied to data using the %>>% operator,
# just as a normal CPO
pp.test = test.data %>>% ret

## ------------------------------------------------------------------------
learner = newcol.cpo %>>% dropcol.cpo %>>%
  impute.cpo %>>% makeLearner("classif.randomForest", 
  predict.type = "prob")

# the new object is a "CPOLearner", subclass of "Learner"
inherits(learner, "CPOLearner")

# train using the task that was not preprocessed
ppmod = train(learner, train.task)

## ----message=FALSE, warning=FALSE----------------------------------------

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

## ------------------------------------------------------------------------

## ------------------------------------------------------------------------
scale = cpoSelect(pattern = "Fare", id = "first") %>>%
  cpoScale(id = "scale")
scale.pca = scale %>>% cpoPca()
cbinder = cpoCbind(scale, scale.pca, cpoSelect(
  pattern = "Age", id = "second"))
result = train.data %>>% cbinder
result[1:3, ]


## ----iml, child="iml.Rnw"------------------------------------------------

## ----imlLibrary, eval=TRUE-----------------------------------------------
library(iml)

## ----mlModel, eval=TRUE--------------------------------------------------
mod

## ----include=FALSE-------------------------------------------------------
train.data = data[rownames(data) %in% train, ]

## ----imlModel, eval=TRUE-------------------------------------------------
X = dropNamed(train.data, "Survived")
iml.mod = Predictor$new(mod, data = X, 
  y = train.data$Survived, class = 2)

## ----featImp, eval=TRUE, fig.height=4, warning=FALSE, message=FALSE, cache=FALSE----
imp = FeatureImp$new(iml.mod, loss = "ce")
plot(imp)

## ----pdp, eval=TRUE, fig.height=3, warning=FALSE, message=FALSE, cache=FALSE----
pdp = PartialDependence$new(iml.mod, feature = "Pclass")
plot(pdp)

## ----lime, eval=TRUE, message=FALSE, fig.height=3, warning=FALSE, message=FALSE, cache=FALSE----
X[1,]

lime = LocalModel$new(iml.mod, x.interest = X[1,])
plot(lime)


## ----openML, child="openml.Rnw"------------------------------------------

## ------------------------------------------------------------------------
library("OpenML")

## ----eval=FALSE----------------------------------------------------------
## setOMLConfig(apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f")
## 
## # Permanently save your API disk to your config file
## saveOMLConfig(apikey = "c1994...47f1f", overwrite=TRUE)

## ----warning=FALSE, message=FALSE----------------------------------------
datasets = listOMLDataSets() 
datasets[1:3, c(1,2,11)]

tasks = listOMLTasks()
tasks[1:3, 1:4]

## ----message=FALSE, warning=FALSE----------------------------------------
listOMLDataSets(data.name = "titanic")[, 1:5]
titanic = getOMLDataSet(data.id = 40945L)

## ----message=FALSE, warning=FALSE----------------------------------------
listOMLTasks(data.name = "titanic")[1:2, 1:4]
titanic.task = getOMLTask(task.id = 146230)
titanic.task

## ----message=FALSE-------------------------------------------------------
lrn = makeLearner("classif.randomForest", mtry = 2)
run.mlr = runTaskMlr(titanic.task, lrn)
run.mlr$bmr$results
# uploadOMLRun(run.mlr)


## ----outlook, child="outlook.Rnw"----------------------------------------



## ----foundation, child="foundation.Rnw"----------------------------------



