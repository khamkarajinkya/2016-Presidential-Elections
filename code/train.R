library(mlr)
library(ranger)
library(knitr)
library(tibble)
library(hot.deck)
library(ggplot2)
df= get(load("Subset1.RData"))
df= data.frame(as_tibble(df))
df1=df[,c("commonweight","gender", 
                     "edloan", 
                     "race",
                     "state",
                     "economyyear",
                     "economynext",
                     "govrating",
                     "grantlegstatus",
                     "deportation",
                     "cleanair",
                     "defensecut",
                     "raisetax",
                     "guncontrol",
                     "medicare",
                     "obamacare",
                     "minwage",
                     "employmentstat",
                     "vote2016")]

#NA Analysis
df.na=data.frame(sapply(df1,function(x)(sum(is.na(x)))))
kable(df.na,format="markdown")

#Hot deck imputation
hd.imp=hot.deck(df1,m=2,method = "p.draw")
final.df=data.frame(hd.imp$data[[1]])
wt=df1$commonweight
final.df$commonweight=NULL
train=data.frame(sapply(final.df,function(x)(factor(x,ordered = FALSE))))

#changing the levels
train$vote=droplevels(train$vote2016)
train$vote2016=NULL

levels(train$vote)=c(levels(train$vote), "Clinton","Trump")

train$vote[train$vote=="Hillary Clinton (Democrat)"]="Clinton"
train$vote[train$vote=="Donald Trump (Republican)"]="Trump"
save(file="train.Rdata",train)

train= data.frame(get(load("train.RData")))


#MlR
param = makeParamSet(
  makeDiscreteParam("num.trees", values = c(300,500,700)),
  makeDiscreteParam("min.node.size", values = c(20,30,50)),
  makeDiscreteParam("mtry",values=c(4,5)))

task=makeClassifTask(id = deparse(substitute(train)), train,"vote",
                     weights = wt, blocking = NULL, positive = NA_character_,
                     fixup.data = "warn", check.data = TRUE)

ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("CV", iters = 5L)


classif.ranger=makeLearner("classif.ranger",predict.type = "prob")

res = tuneParams(classif.ranger, task = task, resampling = rdesc,
                 par.set = param, control = ctrl,measures = list(auc,f1,logloss))

set.seed(1000)


classif.ranger=makeLearner("classif.ranger",predict.type = "prob"
                           ,num.trees=500
                           ,min.node.size=30
                           ,mtry=4)

n = getTaskSize(task)
train.set = sample(n, size = 2*n/3)
test.set = setdiff(1:n, train.set)
train.wt=wt[train.set]

model = mlr::train(classif.ranger, task, subset = train.set,weights = train.wt)
pred1 = predict(model, task = task,subset = test.set)
performance(pred,measures = list(mmce, acc,auc,f1))


#trying gradient boostong model
param = makeParamSet(
  makeDiscreteParam("n.trees", values = c(1000)),
  makeDiscreteParam("shrinkage", values = c(0.01,0.03)),
  makeDiscreteParam("interaction.depth",values=c(2,4,6)),
  makeDiscreteParam("bag.fraction",values=c(0.4,0.6)))

classif.gbm=makeLearner("classif.gbm",predict.type = "prob")

res = tuneParams(classif.gbm, task = task, resampling = rdesc,
                 par.set = param, control = ctrl,measures = list(auc,f1,logloss))

classif.gbm=makeLearner("classif.gbm",predict.type = "prob"
                           ,n.trees=1000
                           ,shrinkage=0.05
                           ,interaction.depth=7
                           ,bag.fraction=0.6
                        )



n = getTaskSize(task)
train.set = sample(n, size = 2*n/3)
test.set = setdiff(1:n, train.set)
train.wt=wt[train.set]

model = mlr::train(classif.gbm, task, subset = train.set,weights = train.wt)

pred2 = predict(model, task = task,subset = test.set)

vote=train$vote
train$vote=NULL
pred_train = data.frame(predict(model, newdata = train))
pred_train$commonweight=wt

final_train=cbind(train,pred_train)

#final model, people who voted for candidates other than Hillary or Trump

df_test= data.frame(get(load("Prediction.RData")))
df1=data.frame(df_test[,c("commonweight","gender", 
                     "edloan", 
                     "race",
                     "state",
                     "economyyear",
                     "economynext",
                     "govrating",
                     "grantlegstatus",
                     "deportation",
                     "cleanair",
                     "defensecut",
                     "raisetax",
                     "medicare",
                     "guncontrol",
                     "obamacare",
                     "minwage",
                     "employmentstat",
                     "vote2016")])

df1$vote2016=NULL
hd.imp=hot.deck(df1,m=1,method = "p.draw")
test=data.frame(hd.imp$data)
wt=test$commonweight
test$commonweight=NULL

pred3 = data.frame(predict(model, newdata = test))
pred3$commonweight=wt

final_test=cbind(test,pred3)

final=rbind(final_train,final_test)

save(final,file = "Prediction.Rdata")

threshold = c(Trump = 0.55, Clinton = 0.45)
pred3 = setThreshold(pred2, threshold = threshold)

performance(pred2,measures = list(mmce, acc,auc,f1))

roc_val = generateThreshVsPerfData(list(RF = pred1, GBM = pred2), measures = list(fpr, tpr))
qplot(x = fpr, y = tpr, color = learner, data = roc_val$data, geom = "path")


plotThreshVsPerf(roc_val)
plotROCCurves(roc_val)

plot(getFeatureImportanceLearner(classif.gbm,model))




#missclassification penalty
costs = matrix(c(0, 4, 6, 0), 2)
colnames(costs) = rownames(costs) = getTaskClassLevels(task)
miss.costs = makeCostMeasure(id = "miss.costs", name = "Miss costs", costs = costs,
                               best = 0, worst = 6)

performance(pred2, measures = list(miss.costs, mmce,auc,f1))

