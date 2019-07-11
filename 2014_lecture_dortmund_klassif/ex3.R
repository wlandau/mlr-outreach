library(methods)
library(mlr)

data("Sonar", package = "mlbench")

# lets try a couple of methods on some tasks

tasks = list(
  makeClassifTask(data = iris, target = "Species"),
  makeClassifTask(data = Sonar, target = "Class")
)

# of course one could change params here
learners = list(
  makeLearner("classif.rpart"),
  makeLearner("classif.randomForest"),
  makeLearner("classif.ksvm")
)

rdesc = makeResampleDesc("CV", iters = 3)

br = benchmark(learners, tasks, rdesc)

# getters for results parts
print(getBMRAggrPerformances(br, as.df = TRUE))
print(head(getBMRPerformances(br, as.df = TRUE)))
print(head(getBMRPredictions(br, as.df = TRUE)))


# remarks:
# wrappers for tuning and feature selection can be used, too (nested sampling)
# their results can of course also be accessed
