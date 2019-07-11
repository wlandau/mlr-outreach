
data = read.csv("titanic.csv", sep = ";")
data = data[-1310,]

library(BBmisc)
data = dropNamed(data, drop = c("boat", "body", "home.dest"))

names(data) = c("Pclass", "Survived", "Name", "Sex",  "Age", "Sibsp", "Parch",
  "Ticket", "Fare", "Cabin", "Embarked")

data$Pclass = as.factor(data$Pclass)
data$Survived = as.factor(data$Survived)
data$Sibsp = as.numeric(data$Sibsp)
data$Parch = as.numeric(data$Parch)
data$Name = as.character(data$Name)

data$Name[data$Name == ""] = NA
data$Name = droplevels(data$Name)

data$Sex[data$Sex == ""] = NA
data$Sex = droplevels(data$Sex)

data$Ticket[data$Ticket == ""] = NA
data$Ticket = droplevels(data$Ticket)

str(data)

getwd()
save(data, file = "data.rda")



