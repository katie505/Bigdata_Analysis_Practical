# 1. 회귀 모형 평가

# 1-1 회귀 모형 평가
## 구축한 회귀 모형에 대한 평가 지표로 (수정된) 결정 계수 이외에도 RMSE, MSE 등이 있음
## RMSE, MSE의 값이 낮을수록 모형의 정확도가 높음
library(ISLR)

hitter <- na.omit(Hitters)

fit_model <- lm(Salary ~ AtBat + Hits + CWalks + Division + PutOuts,
                data = hitters)

second_model <- lm(Salary ~ Hits + CWalks + Division + PutOuts,
                   data = hitters)


library(ModelMetrics)

rmse(fit_model) # 333.9609
mse(fit_model) # 111529.9

rmse(second_model) # 339.1579
mse(second_model) # 115028.1

summary(fit_model)$r.squared # 0.4498717
summary(fit_model)$adj.r.squared # 0.4391688

summary(second_model)$r.squared # 0.4326165
summary(second_model)$adj.r.squared # 0.4238199


# 1-2 분류 모형 평가
## 모형에 대한 평가를 혼동 행렬과 AUC 등을 이용할 수 있음

## 혼동 행렬
library(mlbench)
data('PimaIndiansDiabetes2')
df_pima <- na.omit(PimaIndianDiabetes2)

set.seed(19190301)
train.idx <- sample(1:nrow(df_pima),
                    size = nrow(df_pima) * 0.8)
train <- df_pima[train.idx,]
test <- df_pima[-train.idx,]

library(randomForest)
set.seed(19190301)
md.rf <- randomForest(diabetes ~.,
                      data = train,
                      ntree = 300)
pred <- predict(md.rf, newdata = test)

library(caret)
confusionMatrix(as.factor(pred), test$diabetes)

## AUc : 1과 가까울수록 좋은 모형
auc(actual = test$diabetes,
    predicted = as.factor(pred))
