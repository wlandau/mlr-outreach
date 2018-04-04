#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                              #
#                      DEMO FOR MLR TALK                                       #
#                         OpenML 2018                                          #
#                                                                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Libraries ----
library(mlr)
library(BBmisc)
library(stringi)
library(ggplot2)
library(parallelMap)
library(iml)

# Data ----
train = read.table("train.csv", header = TRUE, sep = ",",
  colClasses = c("integer", "factor", "factor", "character", "factor", 
    "numeric", "numeric", "numeric", "factor", "numeric", "factor", 
    "factor"))
train$train = TRUE

test = read.table("test.csv", header = TRUE, sep = ",",
  colClasses = c("integer", "factor", "character", "factor", "numeric",
    "numeric", "numeric", "factor", "numeric", "factor", "factor"))
test$Survived = NA
test$train = FALSE

data = rbind(train, test)
rm(train, test)

summary(data)

# Feature Engineering ----
# Set empty factor levels to NA
data$Embarked[data$Embarked == ""] = NA
data$Embarked = droplevels(data$Embarked)
data$Cabin[data$Cabin == ""] = NA
data$Cabin = droplevels(data$Cabin)

# Extract possible information from the names
# First split first and last names and save it as a list
names = stri_split(data$Name, fixed = ", ")

# Possible information from the titles of the persons
data$titles = vapply(names, function(name) {
  stri_split(name[2], fixed = " ", simplify = TRUE)[1]
}, character(1))

data$titles = forcats::fct_collapse(data$titles, 
  Noble = c("Capt.", "Col.", "Major.", "Sir.", "Lady.", "Rev.", "Dr.", 
    "Don.", "Dona.", "Jonkheer."),
  # "the" is for "the countess" and got butchered by the splitting earlier
  Mrs = c("Mrs.", "Ms.", "the"), 
  Mr = c("Mr."),
  Miss = c("Mme.", "Mlle.", "Miss."),
  Master = c("Master."))

# "children and women first" we generate a variable to account for this
data$dibs = data$Sex == "female" | data$Age < 15

# Price per person, since multiple ticket prices are bought by one person
data$farePp = data$Fare / (data$Parch + data$SibSp + 1)

# The deck can be extracted from the the cabin number
data$deck = as.factor(stri_sub(data$Cabin, 1, 1))

# Starboard had an odd number, portside even cabin numbers
data$portside = as.numeric(stri_sub(data$Cabin, 3, 3)) %% 2 == 0

# Drop PassengerId, Ticket, Name and Cabin
data = dropNamed(data, c("Cabin","PassengerId", "Ticket", "Name"))

# The deck can be extracted from the the cabin number
data$deck = as.factor(stri_sub(data$Cabin, 1, 1))

# Starboard had an odd number, portside even cabin numbers
data$portside = as.numeric(stri_sub(data$Cabin, 3, 3)) %% 2 == 0

# Drop PassengerId, Ticket, Name and Cabin
data = dropNamed(data, c("Cabin","PassengerId", "Ticket", "Name"))

# Imputation ----
# Remove missing values
data = impute(data, cols = list(
  Age = imputeMedian(),
  Fare = imputeMedian(),
  Embarked = imputeConstant("__miss__"),
  dibs = imputeConstant("__miss__"),
  farePp = imputeMedian(),
  deck = imputeConstant("__miss__"),
  portside = imputeConstant("__miss__")
))

# Task ----
# Split back into training and test data
data = data$data
data = convertDataFrameCols(data, chars.as.factor = TRUE)

train = data[data$train, ]
train$train = NULL

test = data[!data$train, ]
test$train = NULL

task.train = makeClassifTask(id = "titanic", data = train, 
  target = "Survived", positive = "1")

# Learner ----
learners = makeLearners(c("glmnet", "naiveBayes", "randomForest", "ksvm"),
  type = "classif", predict.type = "prob")

set.seed(3)
rdesc = makeResampleDesc("CV", iters = 10L, stratify = TRUE)
rinst = makeResampleInstance(rdesc, task.train)

# Parallelization ----
parallelStart()

# Benchmark ----
bmr = benchmark(learners, task.train, rinst, measures = auc)
parallelStop()

plotBMRBoxplots(bmr)

# Tuning ----
par.set = makeParamSet(
  makeNumericParam("C", lower = -8, upper = 8, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -8, upper = 8, trafo = function(x) 2^x)
)

tune.ctrl = makeTuneControlRandom(maxit = 20L)
classif.ksvm.tuned = makeTuneWrapper(learners$classif.ksvm, resampling = cv3,
  par.set = par.set, control = tune.ctrl)

bmr2 = benchmark(classif.ksvm.tuned, task.train, rinst)
plotBMRBoxplots(mergeBenchmarkResults(list(bmr, bmr2)))

# Performance ----
split = makeResampleInstance(hout, task.train)
mod = train(learners$classif.randomForest, task.train, 
  subset = split$train.inds[[1]])
pred = predict(mod, task.train, subset = split$test.inds[[1]])

df = generateThreshVsPerfData(pred, list(fpr, tpr, acc))

plotROCCurves(df)

calculateROCMeasures(pred)

# IML ----
# Create IML Model
X = train[which(names(train) != "Survived")]
iml.mod = Predictor$new(mod, data = X, y = train$Survived, class = 2)

# Feature Importance
imp = FeatureImp$new(iml.mod, loss = "ce")
plot(imp)

# Partial Dependence Plot
pdp = PartialDependence$new(iml.mod, feature = "Pclass")
plot(pdp)

# LIME
X[1,]

lime = LocalModel$new(iml.mod, x.interest = X[1,], k = 9)
plot(lime)



