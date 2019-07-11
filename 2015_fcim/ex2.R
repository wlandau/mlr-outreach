### Basic resampling and model comparison

library(methods)
library(mlbench)
library(mlr)

# lets use the data from the last lecture tutorial
# and repeat some of our exercises

data = mlbench.threenorm(1000)
task = makeClassifTask(data = as.data.frame(data), target = "classes")
print(task)

# 2 learner to compare, we see how to set hyperpars
lrn1 = makeLearner("classif.kknn", k = 7)
lrn2 = makeLearner("classif.qda")

# cross-validation, we pregenerate the splits for better comparison
rdesc = makeResampleDesc("CV", iters = 10)
rin = makeResampleInstance(rdesc, task = task)

# this compares on the same splits
r1 = resample(lrn1, task, rin)
r2 = resample(lrn2, task, rin)
# peak into object
print(names(r1))
print(r1$measures.test)
print(r1$aggr)
print(head(as.data.frame(r1$pred)))

# this works, too, but uses new splits every time
r1 = resample(lrn1, task, rdesc)
r2 = resample(lrn2, task, rdesc)

# bootstrapping (no big difference in coding...)
rdesc = makeResampleDesc("Bootstrap", iters = 10)
r = resample(lrn1, task, rdesc)

# actuallly we have many resampling methods available
?makeResampleDesc

