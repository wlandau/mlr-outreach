### Probabilities and ROC stuff

library(ROCR)
library(mlr)
library(mlbench)
data(Ionosphere)

# lets use a data set with only 2 classes

task = makeClassifTask(data = Ionosphere, target = "Class", positive = "good")
print(task)

# basic preprocessing
task = removeConstantFeatures(task)


# train + predict
lrn = makeLearner("classif.lda", predict.type = "prob")
mod = train(lrn, task, subset = seq(1, 150, 2))
pred = predict(mod, task, subset = seq(2, 150, 2))
print(pred)

# eval
p = performance(pred, measures = list(mmce, auc))
print(p)

# use different thresholds
pred2 = setThreshold(pred, th = 0.1)
print(getConfMatrix(pred2))
pred2 = setThreshold(pred, th = 0.9)
print(getConfMatrix(pred2))

# convert to ROCR package data format so we can plot
rocr.pred = asROCRPrediction(pred)
rocr.perf = ROCR::performance(rocr.pred, "tpr", "fpr")
plot(rocr.perf)












