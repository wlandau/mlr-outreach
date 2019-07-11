# some benchmark example es before
library(methods)
library(mlr)
library(parallelMap)

data("Sonar", package = "mlbench")

tasks = list(
  makeClassifTask(data = iris, target = "Species"),
  makeClassifTask(data = Sonar, target = "Class")
)

learners = list(
  makeLearner("classif.rpart"),
  makeLearner("classif.randomForest"),
  makeLearner("classif.ksvm")
)

rdesc = makeResampleDesc("CV", iters = 5)

parallelStartSocket(level = "mlr.benchmark")
br = benchmark(learners, tasks, rdesc)
parallelStop()

parallelStartSocket(level = "mlr.resample")
br = benchmark(learners, tasks, rdesc)
parallelStop()

